use std::sync::atomic::{AtomicU32, Ordering};
use crate::network::event::event::Event;

pub struct EventProducer<'a> {
    buf: *mut Event,
    tail: &'a AtomicU32,
    capacity: u32,
}

impl<'a> EventProducer<'a> {
    pub(super) fn new(buf: *mut Event, tail: &'a AtomicU32, capacity: u32) -> Self {
        Self { buf, tail, capacity }
    }

    pub fn push(&self, event: Event) {
        let idx = self.tail.fetch_add(1, Ordering::Relaxed);
        unsafe {
            self.buf.add((idx % self.capacity) as usize).write(event);
        }
    }
}
