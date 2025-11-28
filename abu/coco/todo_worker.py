#!/usr/bin/env python3
"""
Worker module for computing influence scores in parallel.

This script is adapted from AttributeByUnlearning/new_mscoco/todo_worker.py
to work with FastGDA's data structure.

Each worker:
1. Picks up a task (query index) from a shared todo list
2. Computes influence scores for that query
3. Marks the task as complete
4. Repeats until no tasks remain

All required modules (loader, models, unlearning, etc.) are included in this directory.
"""

import os
import sys
import time
import argparse
from filelock import FileLock, Timeout

from process_sample import process_sample


# Global file path variables (will be set based on the provided todo_dir)
TODO_FILE = None
INPROGRESS_FILE = None
COMPLETE_FILE = None

LOCK_TIMEOUT = 120  # seconds to wait for acquiring any file lock


def get_job_name():
    """Return the job name from the environment; default to 'default_job' if not set."""
    return os.environ.get('JOB_NAME', 'default_job')


def initialize_task_dir(task_dir, total_samples):
    """
    If the specified folder does not contain a todo_list.txt file,
    initialize the todo_list.txt with sample indices from 0 to total_samples - 1,
    and create empty inprogress.txt and complete.txt files.
    """
    init_lock_path = os.path.join(task_dir, "init.lock")
    with FileLock(init_lock_path, timeout=LOCK_TIMEOUT):
        files = os.listdir(task_dir)
        if 'todo_list.txt' not in files:
            print(f"Task folder '{task_dir}' is empty. Initializing with {total_samples} tasks.")
            with open(os.path.join(task_dir, 'todo_list.txt'), 'w') as f:
                for i in range(total_samples):
                    f.write(f"{i}\n")
            open(os.path.join(task_dir, 'inprogress.txt'), 'w').close()
            open(os.path.join(task_dir, 'complete.txt'), 'w').close()
        else:
            print(f"Task folder '{task_dir}' already initialized.")


def get_task():
    """
    Atomically retrieves and removes the first sample index from the todo list
    and records it in the inprogress file along with the current job name.
    """
    job_name = get_job_name()
    try:
        with FileLock(TODO_FILE + ".lock", timeout=LOCK_TIMEOUT):
            with FileLock(INPROGRESS_FILE + ".lock", timeout=LOCK_TIMEOUT):
                with open(TODO_FILE, 'r+') as todo_f:
                    todo_lines = todo_f.readlines()
                    if not todo_lines:
                        return None
                    task_line = todo_lines[0].strip()
                    try:
                        sample_idx = int(task_line)
                    except ValueError:
                        print(f"Invalid sample index in todo list: {task_line}")
                        sample_idx = None
                    todo_f.seek(0)
                    todo_f.truncate()
                    todo_f.writelines(todo_lines[1:])
                with open(INPROGRESS_FILE, 'a') as inprog_f:
                    inprog_f.write(f"{sample_idx} {job_name}\n")
                return sample_idx
    except Timeout:
        print("Error: Could not acquire lock on the todo or inprogress file.")
        return None


def complete_task(sample_idx):
    """
    Marks a sample index as complete.
    It removes the task from inprogress.txt and appends it to complete.txt.
    """
    job_name = get_job_name()
    try:
        with FileLock(INPROGRESS_FILE + ".lock", timeout=LOCK_TIMEOUT):
            with open(INPROGRESS_FILE, 'r+') as inprog_f:
                lines = inprog_f.readlines()
                new_lines = []
                found = False
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    idx_str, job = parts[0], parts[1]
                    if int(idx_str) == sample_idx and job == job_name and not found:
                        found = True
                        continue
                    new_lines.append(line)
                inprog_f.seek(0)
                inprog_f.truncate()
                inprog_f.writelines(new_lines)
    except Timeout:
        print("Error: Could not acquire lock on the inprogress file.")

    try:
        with FileLock(COMPLETE_FILE + ".lock", timeout=LOCK_TIMEOUT):
            with open(COMPLETE_FILE, 'a') as comp_f:
                comp_f.write(f"{sample_idx} {job_name}\n")
    except Timeout:
        print("Error: Could not acquire lock on the complete file.")


def requeue_task(sample_idx):
    """
    Requeues a failed task by removing it from inprogress.txt and inserting it at the
    beginning of todo_list.txt (FIFO).
    """
    job_name = get_job_name()
    try:
        with FileLock(TODO_FILE + ".lock", timeout=LOCK_TIMEOUT):
            with FileLock(INPROGRESS_FILE + ".lock", timeout=LOCK_TIMEOUT):
                with open(INPROGRESS_FILE, 'r+') as inprog_f:
                    lines = inprog_f.readlines()
                    new_lines = []
                    found = False
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 2:
                            continue
                        idx_str, job = parts[0], parts[1]
                        if int(idx_str) == sample_idx and job == job_name and not found:
                            found = True
                            continue
                        new_lines.append(line)
                    inprog_f.seek(0)
                    inprog_f.truncate()
                    inprog_f.writelines(new_lines)
                with open(TODO_FILE, 'r+') as todo_f:
                    todo_lines = todo_f.readlines()
                    todo_f.seek(0)
                    new_task_line = f"{sample_idx}\n"
                    todo_f.write(new_task_line)
                    todo_f.writelines(todo_lines)
    except Timeout:
        print("Error: Could not acquire lock when requeuing task.")


def main(args):
    global TODO_FILE, INPROGRESS_FILE, COMPLETE_FILE

    # Ensure the task folder exists
    os.makedirs(args.todo_dir, exist_ok=True)

    # Set file paths based on the provided directory
    TODO_FILE = os.path.join(args.todo_dir, 'todo_list.txt')
    INPROGRESS_FILE = os.path.join(args.todo_dir, 'inprogress.txt')
    COMPLETE_FILE = os.path.join(args.todo_dir, 'complete.txt')

    # Initialize task directory if needed
    initialize_task_dir(args.todo_dir, args.total_samples)

    # Main processing loop
    while True:
        sample_idx = get_task()
        if sample_idx is None:
            print("No tasks available. Exiting.")
            break

        print(f"[{get_job_name()}] Processing sample index: {sample_idx}")
        try:
            # Process the sample using the function from process_sample.py
            process_sample(args, sample_idx_override=sample_idx)
            complete_task(sample_idx)
        except Exception as e:
            print(f"Error processing sample index {sample_idx}: {e}")
            print("Requeuing the sample index.")
            requeue_task(sample_idx)

        # Pause before processing the next task
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Worker for computing influence scores in parallel'
    )
    
    # Required arguments
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--sample_latent_path', type=str, required=True,
                        help='Path to sample latents (.npy)')
    parser.add_argument('--sample_text_path', type=str, required=True,
                        help='Path to sample text embeddings (.npy)')
    parser.add_argument('--sample_root', type=str, required=True,
                        help='Root directory for sample images')
    parser.add_argument('--pretrain_loss_path', type=str, required=True,
                        help='Path to pretrain loss (.npy)')
    parser.add_argument('--todo_dir', type=str, required=True,
                        help='Directory to store todo/inprogress/complete files')
    parser.add_argument('--total_samples', type=int, required=True,
                        help='Total number of sample indices')
    
    # Optional: Nearest neighbor subset (for training)
    parser.add_argument('--nn_pkl', type=str, default=None,
                        help='Path to nearest neighbors pickle (for training subset)')
    parser.add_argument('--nn_num_samples', type=int, default=10000,
                        help='Number of nearest neighbors for subset computation')
    
    # Model and task configuration
    parser.add_argument('--task', type=str, default='mscoco_t2i',
                        help='Task name')
    parser.add_argument('--dataroot', type=str, default='data/coco',
                        help='COCO dataset root directory (should contain trainset/ folder)')
    parser.add_argument('--model_path', type=str, default='data/coco/model.bin',
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--weight_selection', type=str, default='cross-attn-kv',
                        help='Weight selection pattern (cross-attn-kv, cross-attn, attn, all)')
    
    # Fisher matrix configuration
    parser.add_argument('--fisher_type', type=str, default='ekfac',
                        help='Type of Fisher matrix (ekfac)')
    parser.add_argument('--fisher_dir', type=str, default='data/ekfac_fisher',
                        help='Directory containing Fisher matrix')
    
    # Unlearning hyperparameters
    parser.add_argument('--unlearn_lr', type=float, default=0.01,
                        help='Learning rate for unlearning')
    parser.add_argument('--unlearn_steps', type=int, default=1,
                        help='Number of unlearning optimization steps')
    parser.add_argument('--unlearn_batch_size', type=int, default=80,
                        help='Batch size for unlearning')
    parser.add_argument('--unlearn_grad_accum_steps', type=int, default=625,
                        help='Gradient accumulation steps for unlearning')
    
    # Loss computation hyperparameters
    parser.add_argument('--loss_batch_size', type=int, default=8000,
                        help='Batch size for loss calculation')
    parser.add_argument('--loss_time_samples', type=int, default=20,
                        help='Number of time samples to average over')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    main(args)

