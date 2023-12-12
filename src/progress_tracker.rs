pub trait ProgressTracker {
    // new returns a new progress tracker
    fn new(total_items: usize) -> Self;
    // increment_processed increments the progress of total documents processed
    fn increment_processed(&mut self);
    // progress_status returns the current progress status
    fn progress_status(&self) -> (usize, usize);
}
