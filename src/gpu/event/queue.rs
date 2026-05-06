use crate::constants::{T_BETA, H_ALPHA, H_BETA, ALPHA_DECAY, MSLR};
use crate::math::decay::shift_decay_u8;
use std::sync::atomic::{AtomicU32, Ordering};                                                      
                                                                                                    
pub struct EventQueue {
    buf: Box<[Event]>,                                                                             
    tail: AtomicU32,
    head: AtomicU32,
}
                                                                                                    
impl EventQueue {
    pub fn new(capacity: usize) -> Self {    
      let buf = (0..capacity)                                                                        
          .map(|_| Event { event_type: 0, source: 0, timestamp: 0 })
          .collect::<Vec<_>>()                                                                       
          .into_boxed_slice();                                                                     
      Self { buf, tail: AtomicU32::new(0), head: AtomicU32::new(0) }                                 
    }       

    pub fn drain(&self) -> &[Event] {                                                              
        let head = self.head.load(Ordering::Relaxed) as usize;
        let tail = self.tail.load(Ordering::Relaxed) as usize;                                     
        &self.buf[head % self.buf.len()..tail % self.buf.len()]
    }
    
    // returns the raw parts a kernel function needs to push events
    pub fn producer_handle(&self) -> (*mut Event, &AtomicU32, u32) {
        (
            self.buf.as_ptr() as *mut Event,
            &self.tail,
            self.buf.len() as u32,
        )
    }
}