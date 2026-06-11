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

    pub fn drain(&self) -> &[Event] {                                                              
        let head = self.head.load(Ordering::Relaxed) as usize;
        let tail = self.tail.load(Ordering::Relaxed) as usize;                                     
        &self.buf[head % self.buf.len()..tail % self.buf.len()]
    }
    
    pub fn producer_handle(&self) -> EventProducer<'_> {
        EventProducer::new(
            self.buf.as_ptr() as *mut Event,
            &self.tail,
            self.buf.len() as u32,
        )
    }
}