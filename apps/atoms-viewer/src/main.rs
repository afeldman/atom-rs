//! # atoms-viewer (not yet implemented)
//!
//! The interactive desktop application.
//!
//! **Status: scheduled for M2.** Without the `viewer` feature this binary only
//! reports its status and exits, so that a headless `cargo test --workspace`
//! needs neither a GUI toolkit nor a display server.
//!
//! ## Contract (fixed in M0, to be honoured in M2)
//!
//! * The viewer must not require a GPU for scientific calculation: every
//!   quantity shown is computed by the same pure-Rust code path that the tests
//!   and the Python API use.
//! * Long-running calculations must not block the render loop; they run on
//!   worker threads with cancellation and progress reporting.
//! * The first release prioritises a correct, responsive orbital viewer over a
//!   feature-rich interface.
//!
//! ## Planned M2 scope
//!
//! * Orbital explorer: quantum-number selection, isosurface threshold, colour
//!   map, and 2D cross-sections through the 3D density.
//! * Runtime reporting of the selected graphics backend and adapter.

fn main() {
    if cfg!(feature = "viewer") {
        // The `viewer` feature is enabled but the interface is not implemented
        // yet. Fail loudly rather than silently starting an empty window.
        eprintln!(
            "atoms-viewer: the `viewer` feature is enabled but the interface is \
             not implemented yet (scheduled for milestone M2)."
        );
        std::process::exit(2);
    } else {
        println!("atoms-viewer: no GUI in this build.");
        println!("The interactive viewer is scheduled for milestone M2.");
        println!("Rebuild with `--features viewer` once it is implemented.");
    }
}
