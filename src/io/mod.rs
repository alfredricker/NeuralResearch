//! The boundary layer between the network and the outside world.
//!
//! Two symmetric halves:
//!   - afferent (`input`): an external coordinate space (pixels, audio, ...) and the explicit
//!     arrow mapping its coordinates onto *input neurons*. Each input space owns its own
//!     sensory CSR — the map is always visible in code, never implied by allocation order.
//!   - efferent (`output`, future): the arrow from output-neuron spikes back to external
//!     actions / class labels.
//!
//! An input space transduces a frame into events at *encode time* and pushes them into the
//! event queue; the network's own machinery (axon CSR, handlers) carries them from there.

pub mod input;
pub mod output;

pub use input::{InputSpace, SensoryMap, Shape, input_config};
pub use output::{Effector, ReadoutMap, output_config};
