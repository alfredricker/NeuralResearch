/// Need a highly efficient parallel event pusher
/// fetch_add guarantees that each event gets a unique index, and we can write directly to the buffer without locks
use std::sync::atomic::{AtomicU32, Ordering};
use crate::network::event::{Event};
pub unsafe fn push_event(buf: *mut Event, tail: &AtomicU32, capacity: u32, event: Event) {
    let idx = tail.fetch_add(1, Ordering::Relaxed);
    buf.add((idx % capacity) as usize).write(event);
}