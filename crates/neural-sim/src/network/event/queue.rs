use std::sync::atomic::{AtomicU32, Ordering};
use crate::network::event::event::Event;
use crate::network::event::push::EventProducer;

pub struct EventQueue {
    buf: Box<[Event]>,                                                                             
    tail: AtomicU32,
    head: AtomicU32,
}
                                                                                                    
impl EventQueue {
    pub fn new(capacity: usize) -> Self {    
      let buf = (0..capacity)                                                                        
          .map(|_| Event::spike(0, 0, 0))
          .collect::<Vec<_>>()                                                                       
          .into_boxed_slice();                                                                     
      Self { buf, tail: AtomicU32::new(0), head: AtomicU32::new(0) }                                 
    }       

    /// Non-advancing peek at the queued events `[head, tail)`. Used by unit tests to inspect what
    /// a handler pushed. It does NOT recycle slots and its contiguous slice cannot represent a
    /// wavefront that wraps the ring end — so it is only valid at low, pre-wrap indices. The event
    /// loop must use [`next_wavefront`](Self::next_wavefront) instead.
    pub fn drain(&self) -> &[Event] {
        let head = self.head.load(Ordering::Relaxed) as usize;
        let tail = self.tail.load(Ordering::Relaxed) as usize;
        &self.buf[head % self.buf.len()..tail % self.buf.len()]
    }

    /// Take the current **wavefront**: the events queued at call time, `[head, tail)`. Iterating
    /// the returned [`Wavefront`] yields those events by value and advances `head` past each one,
    /// recycling its ring slot. Events that handlers push *during* processing land beyond the
    /// captured `tail`, so they are excluded here and form the *next* wavefront — which is exactly
    /// how one cascade generation advances per call (see `network::event::r#loop::run_event_loop`).
    /// Wrap-around is handled by index arithmetic, so unlike [`drain`](Self::drain) this is correct
    /// at any ring position.
    pub fn next_wavefront(&self) -> Wavefront<'_> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        Wavefront { buf: &self.buf, head: &self.head, cursor: head, end: tail }
    }

    pub fn producer_handle(&self) -> EventProducer<'_> {
        EventProducer::new(
            self.buf.as_ptr() as *mut Event,
            &self.tail,
            self.buf.len() as u32,
        )
    }
}

/// One drained wavefront: an iterator over the captured event range `[cursor, end)`, yielding
/// each [`Event`] by value (the buffer holds `Copy` events) and advancing the queue's `head` as it
/// goes so consumed slots are recycled immediately. `cursor`/`end` are the queue's monotonic u32
/// counters at capture time; the buffer is indexed `cursor % len`, so a wavefront straddling the
/// ring end reads correctly and in order.
pub struct Wavefront<'a> {
    buf: &'a [Event],
    head: &'a AtomicU32,
    cursor: u32,
    end: u32,
}

impl<'a> Iterator for Wavefront<'a> {
    type Item = Event;

    #[inline]
    fn next(&mut self) -> Option<Event> {
        if self.cursor == self.end {
            return None;
        }
        let e = self.buf[(self.cursor % self.buf.len() as u32) as usize];
        self.cursor = self.cursor.wrapping_add(1);
        // publish the advance so the producer may reclaim the slot we just copied out.
        self.head.store(self.cursor, Ordering::Relaxed);
        Some(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::event::event::SOMATIC_SPIKE;

    #[test]
    fn next_wavefront_advances_head_and_isolates_generations() {
        let q = EventQueue::new(16);
        let p = q.producer_handle();
        p.push(Event::spike(SOMATIC_SPIKE, 1, 0));
        p.push(Event::spike(SOMATIC_SPIKE, 2, 0));

        // the first wavefront is exactly the two events queued at call time
        let w1: Vec<u32> = q.next_wavefront().map(|e| e.source).collect();
        assert_eq!(w1, vec![1, 2]);

        // head advanced past them — re-draining now yields nothing
        assert_eq!(q.next_wavefront().count(), 0);

        // events pushed afterwards form the next wavefront, in isolation
        p.push(Event::spike(SOMATIC_SPIKE, 3, 0));
        let w2: Vec<u32> = q.next_wavefront().map(|e| e.source).collect();
        assert_eq!(w2, vec![3]);
    }

    #[test]
    fn wavefront_reads_correctly_across_the_ring_wrap() {
        let q = EventQueue::new(4); // tiny ring to force a wrap
        let p = q.producer_handle();
        for i in 0..3 {
            p.push(Event::spike(SOMATIC_SPIKE, i, 0));
        }
        assert_eq!(q.next_wavefront().count(), 3); // consume → head = 3

        // next three pushes land in slots 3, 0, 1 (wrapping); reading [3, 6) must un-wrap them
        for i in 10..13 {
            p.push(Event::spike(SOMATIC_SPIKE, i, 0));
        }
        let w: Vec<u32> = q.next_wavefront().map(|e| e.source).collect();
        assert_eq!(w, vec![10, 11, 12]);
    }
}