"""
EventLapse Reference Generator: Task 1 Event Counting
Bouncing ball between walls at a fixed 1 contact/sec contact rate.
"""
from eventlapse.generation.event_counting import EventCountingScene, EventCountingGenerator

if __name__ == "__main__":
    gen = EventCountingGenerator()
    print("Task 1 Event Counting Generator initialized successfully.")
