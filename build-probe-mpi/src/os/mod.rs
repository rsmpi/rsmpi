#[cfg(windows)]
mod windows;

#[cfg(unix)]
mod unix;

#[cfg(unix)]
pub use self::unix::probe;
#[cfg(windows)]
pub use self::windows::probe;
