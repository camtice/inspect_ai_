import time
import os
from collections import defaultdict

# Ensure timing display is enabled
os.environ["INSPECT_DISPLAY"] = "true"

# Create a TimerManager to collect and report timing statistics
class TimerManager:
    """Manages timing data collection and reporting."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_time = None
        self.end_time = None
        
    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.start_time = None
        self.end_time = None
        
    def start_generation(self):
        """Mark the start of a generation sequence."""
        self.reset()
        self.start_time = time.perf_counter()
        
    def end_generation(self):
        """Mark the end of a generation sequence."""
        self.end_time = time.perf_counter()
        
    def add_timing(self, name: str, elapsed: float):
        """Add a timing measurement."""
        self.timings[name].append(elapsed)
        
    def get_summary(self):
        """Get a summary of all timing data."""
        summary = {}
        
        # Calculate total time if we have start and end times
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            summary["total_generation_time"] = total_time
        
        # Process individual timing categories
        for name, times in self.timings.items():
            if times:
                category_summary = {
                    "total": sum(times),
                    "count": len(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times)
                }
                
                # Calculate percentage of total time if available
                if "total_generation_time" in summary:
                    category_summary["percent"] = (category_summary["total"] / summary["total_generation_time"]) * 100
                
                summary[name] = category_summary
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of timing data."""
        summary = self.get_summary()
        if not summary:
            print("\nNo timing data collected.")
            return
        
        # Extract total generation time if available
        total_time = summary.get("total_generation_time", 0)
        
        print("\n" + "=" * 70)
        print(f"PERFORMANCE TIMING SUMMARY")
        print("=" * 70)
        
        if total_time:
            print(f"Total generation time: {total_time:.4f} seconds")
            print("-" * 70)
        
        # Group timings by major categories
        major_categories = {
            "Initialization": ["vLLM model initialization", "Model loading"],
            "LoRA Operations": ["Get target modules", "LoRA config creation", "Get PEFT model", 
                               "Injecting noise into LoRA weights", "Saving LoRA adapter"],
            "Noise Injection": ["Adding noise to all weights", "Adding noise to percentage of weights", "Injecting noise"],
            "Generation Steps": ["Chat formatting", "Tokenization", "Generation", "LogProbs computation", "Decoding"],
            "HF Operations": ["HF model generation"],
            "vLLM Operations": ["vLLM generation"],
            "Test Operations": ["Operation 1", "Operation 2", "Operation 3", "Tokenization", "Generation"]
        }
        
        # Process each category
        for category_name, timing_keys in major_categories.items():
            category_data = {k: v for k, v in summary.items() if k in timing_keys and k in summary}
            
            if category_data:
                print(f"\n{category_name}:")
                print("-" * 70)
                
                # Sort by total time (descending)
                sorted_items = sorted(category_data.items(), key=lambda x: x[1]["total"], reverse=True)
                
                # Calculate category totals
                category_total = sum(item[1]["total"] for item in sorted_items)
                category_percent = (category_total / total_time * 100) if total_time else 0
                
                # Print each timing in this category
                for name, data in sorted_items:
                    percent = data.get("percent", 0)
                    print(f"  {name:<30} {data['total']:.4f}s ({percent:.1f}% of total)")
                    
                # Print category summary
                print(f"  {'Category Total:':<30} {category_total:.4f}s ({category_percent:.1f}% of total)")
        
        # Show uncategorized timings
        all_categorized = [item for sublist in major_categories.values() for item in sublist]
        uncategorized = {k: v for k, v in summary.items() 
                         if k not in all_categorized and k != "total_generation_time"}
        
        if uncategorized:
            print("\nOther Operations:")
            print("-" * 70)
            sorted_items = sorted(uncategorized.items(), key=lambda x: x[1]["total"], reverse=True)
            
            for name, data in sorted_items:
                percent = data.get("percent", 0)
                print(f"  {name:<30} {data['total']:.4f}s ({percent:.1f}% of total)")
        
        print("=" * 70 + "\n")


# Create a global TimerManager instance
timer_manager = TimerManager()


# Add a Timer utility class
class Timer:
    """Timer for measuring execution time."""
    
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        elapsed_time = time.perf_counter() - self.start_time
        # Add timing to manager
        timer_manager.add_timing(self.name, elapsed_time)
        # Print individual timing
        print(f"[TIMER] {self.name}: {elapsed_time:.4f} seconds")


# Test function for the timing functionality
def test_timing():
    """Run a simple test of the timing functionality."""
    print("\nRunning timing test...")
    
    # Reset and start timing
    timer_manager.start_generation()
    
    # Simulate some operations
    with Timer("Operation 1"):
        time.sleep(0.1)  # Simulate a short operation
    
    with Timer("Operation 2"):
        time.sleep(0.2)  # Simulate a medium operation
    
    with Timer("Operation 3"):
        time.sleep(0.3)  # Simulate a longer operation
    
    # Simulate a categorized operation
    with Timer("Tokenization"):
        time.sleep(0.15)  # Simulate tokenization
    
    with Timer("Generation"):
        time.sleep(0.25)  # Simulate generation
    
    # End timing and print summary
    timer_manager.end_generation()
    timer_manager.print_summary()
    
    print("Timing test completed.")


# Run the test if this file is executed directly
if __name__ == "__main__":
    test_timing() 