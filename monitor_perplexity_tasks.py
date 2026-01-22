#!/usr/bin/env python3
"""
Perplexity Task Monitoring Dashboard
Monitors and displays execution status of Perplexity AI-based notebook generation
"""

import json
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
from collections import defaultdict

class PerplexityTaskMonitor:
    """
    Monitor for tracking Perplexity task execution status and metrics
    """
    
    def __init__(self, log_file='perplexity_task_logs.json'):
        self.log_file = log_file
        self.logs = self.load_logs()
    
    def load_logs(self):
        """
        Load all execution logs from JSON file
        """
        logs = []
        
        if not os.path.exists(self.log_file):
            return logs
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            log_entry = json.loads(line)
                            logs.append(log_entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error loading logs: {str(e)}")
        
        return logs
    
    def get_last_execution(self):
        """
        Get the most recent task execution
        """
        if not self.logs:
            return None
        
        return self.logs[-1]
    
    def get_success_rate(self, days=7):
        """
        Calculate success rate for last N days
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Success rate percentage (0-100)
        """
        if not self.logs:
            return 0
        
        threshold = datetime.now() - timedelta(days=days)
        successful = 0
        total = 0
        
        for log_entry in self.logs:
            try:
                timestamp = datetime.fromisoformat(log_entry['timestamp'])
                
                if timestamp > threshold:
                    total += 1
                    if log_entry.get('success'):
                        successful += 1
            except (KeyError, ValueError):
                continue
        
        return (successful / total * 100) if total > 0 else 0
    
    def get_failure_reasons(self, days=7):
        """
        Get all failure reasons for last N days
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with failure counts
        """
        threshold = datetime.now() - timedelta(days=days)
        failures = defaultdict(int)
        
        for log_entry in self.logs:
            try:
                timestamp = datetime.fromisoformat(log_entry['timestamp'])
                
                if timestamp > threshold and not log_entry.get('success'):
                    error = log_entry.get('error', 'Unknown error')
                    # Get first line of error for grouping
                    error_key = error.split('\n')[0][:60]
                    failures[error_key] += 1
            except (KeyError, ValueError):
                continue
        
        return dict(failures)
    
    def get_execution_times(self, days=7):
        """
        Get execution time statistics for successful runs
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with time statistics
        """
        threshold = datetime.now() - timedelta(days=days)
        execution_times = []
        
        for log_entry in self.logs:
            try:
                timestamp = datetime.fromisoformat(log_entry['timestamp'])
                
                if timestamp > threshold and log_entry.get('success'):
                    result = log_entry.get('result', {})
                    if isinstance(result, dict) and 'execution_time' in result:
                        execution_times.append(result['execution_time'])
            except (KeyError, ValueError, TypeError):
                continue
        
        if not execution_times:
            return {}
        
        return {
            'min': min(execution_times),
            'max': max(execution_times),
            'avg': sum(execution_times) / len(execution_times),
            'count': len(execution_times)
        }
    
    def get_consecutive_successes(self):
        """
        Get count of consecutive successful executions
        
        Returns:
            Number of consecutive successful runs from most recent
        """
        if not self.logs:
            return 0
        
        count = 0
        for log_entry in reversed(self.logs):
            if log_entry.get('success'):
                count += 1
            else:
                break
        
        return count
    
    def get_statistics(self, days=7):
        """
        Get comprehensive statistics for the monitoring period
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with complete statistics
        """
        return {
            'total_executions': len([l for l in self.logs if self.is_in_last_days(l, days)]),
            'success_rate': self.get_success_rate(days),
            'consecutive_successes': self.get_consecutive_successes(),
            'failure_reasons': self.get_failure_reasons(days),
            'execution_times': self.get_execution_times(days),
            'last_execution': self.get_last_execution(),
            'analysis_period_days': days
        }
    
    def is_in_last_days(self, log_entry, days):
        """
        Check if log entry is within last N days
        """
        try:
            timestamp = datetime.fromisoformat(log_entry['timestamp'])
            threshold = datetime.now() - timedelta(days=days)
            return timestamp > threshold
        except (KeyError, ValueError):
            return False
    
    def print_status(self, days=7):
        """
        Print formatted status report
        
        Args:
            days: Number of days to analyze
        """
        stats = self.get_statistics(days)
        last_exec = stats['last_execution']
        
        print("\n" + "="*70)
        print(" " * 15 + "PERPLEXITY TASK MONITORING DASHBOARD")
        print("="*70)
        
        # Last Execution Info
        print("\n[LAST EXECUTION]")
        if last_exec:
            print(f"  Time: {last_exec.get('timestamp', 'N/A')}")
            print(f"  Status: {'✓ SUCCESS' if last_exec.get('success') else '✗ FAILED'}")
            if not last_exec.get('success'):
                error = last_exec.get('error', 'Unknown error')
                print(f"  Error: {error[:100]}...")
            print(f"  Execution ID: {last_exec.get('execution_id', 'N/A')}")
        else:
            print("  No executions recorded")
        
        # Statistics
        print(f"\n[STATISTICS - LAST {days} DAYS]")
        print(f"  Total Executions: {stats['total_executions']}")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print(f"  Consecutive Successes: {stats['consecutive_successes']}")
        
        # Execution Times
        exec_times = stats['execution_times']
        if exec_times:
            print(f"\n[EXECUTION TIME STATISTICS]")
            print(f"  Minimum: {exec_times['min']:.2f} seconds")
            print(f"  Maximum: {exec_times['max']:.2f} seconds")
            print(f"  Average: {exec_times['avg']:.2f} seconds")
            print(f"  Successful Runs: {exec_times['count']}")
        
        # Failures
        failures = stats['failure_reasons']
        if failures:
            print(f"\n[RECENT FAILURES]")
            for error, count in sorted(failures.items(), key=lambda x: x[1], reverse=True):
                print(f"  {count}x - {error}")
        else:
            print(f"\n[FAILURES]")
            print("  No failures recorded!")
        
        print("\n" + "="*70 + "\n")
    
    def print_json(self, days=7):
        """
        Print statistics as JSON
        """
        stats = self.get_statistics(days)
        print(json.dumps(stats, indent=2, default=str))
    
    def export_report(self, filename='perplexity_task_report.json', days=7):
        """
        Export monitoring report to JSON file
        
        Args:
            filename: Output filename
            days: Number of days to analyze
        """
        stats = self.get_statistics(days)
        stats['generated_at'] = datetime.now().isoformat()
        
        try:
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"Report exported to {filename}")
        except Exception as e:
            print(f"Error exporting report: {str(e)}")


def main():
    """
    Main entry point for the monitoring dashboard
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Perplexity Task Monitoring Dashboard'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to analyze (default: 7)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of formatted text'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export report to JSON file with specified name'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='perplexity_task_logs.json',
        help='Path to log file (default: perplexity_task_logs.json)'
    )
    
    args = parser.parse_args()
    
    # Create monitor instance
    monitor = PerplexityTaskMonitor(log_file=args.log_file)
    
    # Output in requested format
    if args.export:
        monitor.export_report(filename=args.export, days=args.days)
    elif args.json:
        monitor.print_json(days=args.days)
    else:
        monitor.print_status(days=args.days)


if __name__ == '__main__':
    main()
