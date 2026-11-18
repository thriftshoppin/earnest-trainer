"""Visible local Earnest training window. Shows the stack being built (stacked.py) when one is named or
exists, otherwise the TinyStories baseline run. Reads saved run files; training runs as a child of this window
and stops when the window is closed."""
import argparse
import ctypes
import feedback
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A writer may be in the middle of the last line.
    return result


def tail(path, count=7):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 4000))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    return '\n'.join(line[:150] for line in lines[-count:])


def keep_awake(on):
    """Stop the computer and screen from sleeping while a training child runs."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000003 if on else 0x80000000)
    except Exception:
        pass


def stack_state(stack):
    """Everything the window shows about a stack, from its saved files."""
    meta = json.loads((stack / 'stack.json').read_text(encoding='utf-8'))
    blocks = meta['blocks']
    points, boundaries, offset = [], [], 0
    samples, sample_at, stage_step, speed, train_bpb, newest = None, None, 0, None, None, 0.0
    for index in range(len(blocks) + 1):
        folder = stack / f'stage-{index:03d}'
        evaluations = rows(folder / 'evaluations.jsonl')
        metrics = rows(folder / 'metrics.jsonl')
        for row in evaluations:
            points.append({'step': offset + row['step'], 'validation_bpb': row['validation_bpb']})
            if row.get('samples'):
                samples, sample_at = row['samples'], (index, row['step'])
        last = max([row['step'] for row in evaluations + metrics] + [0])
        if index < len(blocks):
            offset += last
            boundaries.append(offset)
        else:
            stage_step = last
            if metrics:
                train_bpb = metrics[-1]['train_bpb']
            if len(metrics) >= 2 and metrics[-1]['elapsed_seconds'] > metrics[-2]['elapsed_seconds']:
                speed = ((metrics[-1]['elapsed_seconds'] - metrics[-2]['elapsed_seconds'])
                         / max(1, metrics[-1]['step'] - metrics[-2]['step']))
            for name in ('latest.pt', 'metrics.jsonl', 'evaluations.jsonl'):
                if (folder / name).exists():
                    newest = max(newest, (folder / name).stat().st_mtime)
    crawl_review = None
    if 'crawl' in meta:
        reports = list(stack.glob('stage-*/crawl-review-*.json'))
        if reports:
            report = max(reports, key=lambda path: path.stat().st_mtime_ns)
            crawl_review = json.loads(report.read_text(encoding='utf-8'))
            crawl_review['key'] = (str(report), report.stat().st_mtime_ns)
    return {'width': meta['config']['width'], 'context': meta['config']['context'], 'frozen': len(blocks),
            'is_crawl': 'crawl' in meta, 'crawl_review': crawl_review,
            'parameters': sum(block['parameters'] for block in blocks), 'stage': len(blocks),
            'stage_step': stage_step, 'total_steps': offset + stage_step, 'points': points,
            'boundaries': boundaries, 'validation': points[-1]['validation_bpb'] if points else None,
            'train_bpb': train_bpb, 'seconds_per_step': speed, 'samples': samples, 'sample_at': sample_at,
            'fresh': time.time() - newest < 120}


class Monitor:
    def __init__(self, options):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.options = tk, options
        self.stack = options.stack
        self.window = tk.Tk()
        self.window.title('Earnest — Local Training')
        screen_w, screen_h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        width, height = min(940, screen_w - 40), min(780, screen_h - 90)
        self.window.geometry(f'{width}x{height}+{(screen_w - width) // 2}+{max(0, (screen_h - height) // 2 - 25)}')
        self.window.minsize(700, 480)
        self.child = None
        self.log = None
        self.stopped = False
        self.stop_file = None
        self.target = 1000
        self.folder = None
        self.last_sample = None
        self.prompt_window = None
        self.prompt_running = False
        style = ttk.Style()
        style.theme_use('clam')
        frame = ttk.Frame(self.window, padding=16)
        frame.pack(fill='both', expand=True)
        self.saved = tk.StringVar()
        saved_label = ttk.Label(frame, textvariable=self.saved, wraplength=width - 50)
        bar = ttk.Frame(frame)
        bar.pack(side='bottom', fill='x', pady=(4, 0))
        saved_label.pack(side='bottom', anchor='w', pady=2)
        if self.stack:
            self.continue_button = ttk.Button(bar, text='Start building', command=self.start_stack)
            self.continue_button.pack(side='left')
            self.stop_button = ttk.Button(bar, text='Stop', command=self.stop, state='disabled')
            self.stop_button.pack(side='left', padx=8)
            ttk.Button(bar, text='Open stack folder', command=self.open_folder).pack(side='left')
            ttk.Button(bar, text='Test prompt', command=self.open_prompt_test).pack(side='left', padx=8)
            self.folder = self.stack
        else:
            self.continue_button = ttk.Button(bar, text='Continue for 1,000 updates', command=self.continue_run)
            self.continue_button.pack(side='left')
            ttk.Button(bar, text='Open run folder', command=self.open_folder).pack(side='left', padx=8)
            ttk.Button(bar, text='Open report', command=self.open_report).pack(side='left')
        ttk.Label(frame, text='EARNEST', font=('Segoe UI', 22, 'bold')).pack(anchor='w')
        subtitle = (f'Stack "{self.stack.name}" • one block at a time • CPU training • every block saved on the drive'
                    if self.stack else 'TinyStories baseline • CPU training • checkpoints saved locally')
        ttk.Label(frame, text=subtitle, font=('Segoe UI', 11)).pack(anchor='w', pady=(0, 8))
        self.status = tk.StringVar(value='Reading run…')
        ttk.Label(frame, textvariable=self.status, font=('Segoe UI', 14, 'bold')).pack(anchor='w')
        self.detail = tk.StringVar()
        ttk.Label(frame, textvariable=self.detail, font=('Segoe UI', 11), wraplength=width - 50).pack(anchor='w', pady=4)
        self.progress = ttk.Progressbar(frame, maximum=1000)
        self.progress.pack(fill='x', pady=4)
        self.chart = tk.Canvas(frame, height=170, bg='#14202b', highlightthickness=0)
        self.chart.pack(fill='x', pady=6)
        self.sample_title = tk.StringVar(value='Fixed-prompt samples from the latest evaluation')
        ttk.Label(frame, textvariable=self.sample_title, font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        sample_frame = ttk.Frame(frame)
        sample_frame.pack(fill='both', expand=True, pady=4)
        self.sample = tk.Text(sample_frame, height=9, wrap='word', font=('Consolas', 10), bg='#f4f6f8')
        sample_scroll = ttk.Scrollbar(sample_frame, orient='vertical', command=self.sample.yview)
        self.sample.configure(yscrollcommand=sample_scroll.set)
        sample_scroll.pack(side='right', fill='y')
        self.sample.pack(side='left', fill='both', expand=True)
        self.feed = None
        if self.stack:
            ttk.Label(frame, text='Trainer output', font=('Segoe UI', 11, 'bold')).pack(anchor='w')
            self.feed = tk.Text(frame, height=6, wrap='none', font=('Consolas', 9), bg='#14202b', fg='#dce8f2')
            self.feed.pack(fill='x', pady=4)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after(2500, lambda: self.window.attributes('-topmost', False))
        self.window.focus_force()
        if self.stack and options.start:
            self.window.after(500, self.start_stack)
        self.refresh()
        self.window.mainloop()

    # ------------------------------------------------------------------ shared

    def open_folder(self):
        if self.folder and Path(self.folder).exists():
            os.startfile(self.folder)

    def open_report(self):
        if self.folder and (self.folder / 'REPORT.md').exists():
            os.startfile(self.folder / 'REPORT.md')

    def open_prompt_test(self):
        from tkinter import ttk

        if self.prompt_window is not None and self.prompt_window.winfo_exists():
            self.prompt_window.lift()
            self.prompt_window.focus_force()
            return

        tk = self.tk
        dialog = self.prompt_window = tk.Toplevel(self.window)
        dialog.title(f'Test prompt — {self.stack.name}')
        dialog.minsize(500, 400)
        panel = ttk.Frame(dialog, padding=12)
        panel.pack(fill='both', expand=True)
        ttk.Label(panel, text='Prompt').pack(anchor='w')
        prompt = tk.Text(panel, height=5, width=64, wrap='word', font=('Consolas', 11))
        prompt.pack(fill='x', pady=(2, 8))
        meta_path = self.stack / 'stack.json'
        stack_meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        observation_only = 'crawl' in stack_meta
        prompt.insert('1.0', 'The sun is' if observation_only else 'Once upon a time,')
        controls = ttk.Frame(panel)
        controls.pack(fill='x')
        ttk.Label(controls, text='Seed').pack(side='left')
        seed = ttk.Entry(controls, width=10)
        seed.insert(0, '2026')
        seed.pack(side='left', padx=6)
        status = tk.StringVar(value='Uses the frozen stack and generates up to 120 bytes per prompt.')
        ttk.Label(panel, textvariable=status, wraplength=550).pack(anchor='w', pady=8)
        ttk.Label(panel, text='Continuation').pack(anchor='w')
        output = tk.Text(panel, height=8, width=64, wrap='word', font=('Consolas', 11), state='disabled')
        output.pack(fill='both', expand=True, pady=(2, 0))
        results = queue.SimpleQueue()
        draft = None
        ttk.Label(panel, text='Corrected continuation').pack(anchor='w', pady=(8, 0))
        correction = tk.Text(panel, height=3, width=64, wrap='word')
        correction.pack(fill='x', pady=(2, 5))
        feedback_bar = ttk.Frame(panel)
        feedback_bar.pack(fill='x')

        def save_feedback(corrected):
            nonlocal draft
            if draft is None:
                return
            if prompt.get('1.0', 'end-1c') != draft['prompt']:
                status.set('The prompt changed. Generate a new draft first.')
                return
            approved = correction.get('1.0', 'end-1c') if corrected else draft['response']
            try:
                feedback.record(self.stack, draft, approved)
            except (OSError, ValueError) as error:
                status.set(str(error))
                return
            status.set('Saved for the next training round. New draft uses a different seed.')
            draft = None
            accept.configure(state='disabled')
            correct.configure(state='disabled')

        accept = ttk.Button(feedback_bar, text='Accept', command=lambda: save_feedback(False), state='disabled')
        accept.pack(side='left')
        correct = ttk.Button(feedback_bar, text='Correct', command=lambda: save_feedback(True), state='disabled')
        correct.pack(side='left', padx=6)
        if observation_only:
            for widget in panel.pack_slaves()[-3:]:
                widget.pack_forget()
            status.set('Phase 0: raw generation from the active checkpoint. Observation only.')

        def new_draft():
            try:
                value = (int(seed.get()) + 1) % 2**64
            except ValueError:
                value = 2026
            seed.delete(0, 'end')
            seed.insert(0, str(value))
            generate()

        def generate():
            nonlocal draft
            if self.prompt_running:
                status.set('A prompt test is already running.')
                return
            content = prompt.get('1.0', 'end-1c')
            if not content.strip():
                status.set('Enter a prompt first.')
                return
            try:
                chosen_seed = int(seed.get())
                if not 0 <= chosen_seed < 2**64:
                    raise ValueError
            except ValueError:
                status.set('Enter a whole-number seed from 0 to 2^64−1.')
                return
            status.set('Generating…')
            draft = None
            correction.delete('1.0', 'end')
            accept.configure(state='disabled')
            correct.configure(state='disabled')
            self.set_text(output, '')
            button.configure(state='disabled')
            self.prompt_running = True

            def worker():
                command = [str(PYTHON), '-B', str(ROOT / 'prompt_probe.py'), '--stack',
                           str(self.stack), '--seed', str(chosen_seed)]
                try:
                    run = subprocess.run(command, cwd=ROOT, input=content.encode('utf-8'),
                                         capture_output=True, timeout=90,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                    if run.returncode:
                        results.put((None, run.stderr.decode('utf-8', errors='replace').strip().splitlines()[-1]
                                     if run.stderr.strip() else 'The prompt tester exited unexpectedly.'))
                    else:
                        results.put((json.loads(run.stdout), None))
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    results.put((None, str(error)))

            def poll():
                nonlocal draft
                try:
                    result, error = results.get_nowait()
                except queue.Empty:
                    self.window.after(100, poll)
                    return
                self.prompt_running = False
                if not dialog.winfo_exists():
                    return
                button.configure(state='normal')
                if error:
                    status.set(error)
                else:
                    draft = {**result, 'prompt': content}
                    if not observation_only:
                        accept.configure(state='normal')
                        correct.configure(state='normal')
                    self.set_text(output, result['response'])
                    status.set(f'{result["blocks"]} frozen blocks · seed {result["seed"]}')

            threading.Thread(target=worker, daemon=True).start()
            self.window.after(100, poll)

        button = ttk.Button(controls, text='Generate', command=generate)
        button.pack(side='left', padx=6)
        ttk.Button(controls, text='New draft', command=new_draft).pack(side='left')
        prompt.focus_set()

    def running(self):
        return self.child is not None and self.child.poll() is None

    def stop(self):
        if self.running():
            self.stopped = True
            if self.stop_file is not None:
                self.stop_file.touch()
                return
            self.child.terminate()  # legacy jobs save every 50 updates
        keep_awake(False)

    def close(self):
        self.stop()
        if self.stop_file is not None and self.running():
            self.window.after(100, self.close)
            return
        self.window.destroy()

    def set_text(self, widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def draw_chart(self, evaluations, boundaries=(), skip_first=False):
        self.chart.delete('all')
        width = max(self.chart.winfo_width(), 660)
        self.chart.create_text(14, 12, text='VALIDATION BITS / BYTE', fill='#dce8f2', anchor='nw')
        if not evaluations:
            return
        shown = evaluations[1:] if skip_first and len(evaluations) > 3 else evaluations
        ymax = max(row['validation_bpb'] for row in shown) * 1.05
        ymin = min(row['validation_bpb'] for row in shown) * 0.95
        xmax = max(1, max(row['step'] for row in shown))
        span = max(ymax - ymin, 1e-6)

        def place(row):
            return 50 + (width - 90) * row['step'] / xmax, 145 - 105 * (row['validation_bpb'] - ymin) / span

        for boundary in boundaries:
            x = 50 + (width - 90) * boundary / xmax
            self.chart.create_line(x, 30, x, 150, fill='#3b556b', dash=(3, 3))
        coordinates = [value for row in shown for value in place(row)]
        if len(coordinates) >= 4:
            self.chart.create_line(*coordinates, fill='#67e2ae', width=2)
        labelled = shown if len(shown) <= 12 else [shown[0], shown[-1]]
        for row in labelled:
            x, y = place(row)
            self.chart.create_oval(x - 3, y - 3, x + 3, y + 3, fill='#67e2ae', outline='')
            self.chart.create_text(x, y - 11, text=f"{row['validation_bpb']:.2f}", fill='white')
        self.chart.create_text(50, 160, text='Update 0', fill='#dce8f2')
        self.chart.create_text(width - 50, 160, text=f'Update {xmax:,}', fill='#dce8f2')

    def refresh(self):
        try:
            if self.stack:
                self.refresh_stack()
            else:
                self.refresh_baseline()
        except (OSError, ValueError, KeyError) as error:
            self.status.set(f'Waiting for saved state: {error}')
        self.window.after(1000, self.refresh)

    # ------------------------------------------------------------------ stack

    def start_stack(self):
        if self.running():
            return
        options = self.options
        self.log = (ROOT / 'stacks' / f'{self.stack.name}-console.log').open('a', encoding='utf-8')
        if not (self.stack / 'stack.json').exists():
            created = subprocess.run([str(PYTHON), str(ROOT / 'stacked.py'), 'new', '--stack', str(self.stack),
                                      '--width', str(options.width), '--train-bytes', str(options.train_bytes)],
                                     cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            if created.returncode != 0:
                self.status.set('Could not create the stack — see trainer output')
                return
        self.stopped = False
        command = [str(PYTHON), '-B', '-u', str(ROOT / 'stacked.py')]
        meta = json.loads((self.stack / 'stack.json').read_text(encoding='utf-8'))
        self.stop_file = None
        if 'crawl' in meta:
            self.stop_file = self.stack / f'.stop-{uuid.uuid4().hex}'
            command += ['grow', '--stack', str(self.stack), '--blocks', str(options.blocks),
                        '--continuous', '--round-steps', str(options.train_only_steps or 400),
                        '--threads', '1', '--stop-file', str(self.stop_file)]
        elif options.train_only_steps:
            command += ['train', '--stack', str(self.stack), '--steps', str(options.train_only_steps),
                        '--max-minutes', str(options.hours * 60), '--threads', '1']
        else:
            command += ['grow', '--stack', str(self.stack), '--blocks', str(options.blocks),
                        '--hours', str(options.hours), '--threads', str(options.threads)] + (['--no-cache'] if options.no_cache else [])
            if options.continuous:
                command += ['--continuous', '--round-steps', '1000', '--stage-steps', '2000',
                            '--lr', '0.0005', '--threads', '1']
        self.child = subprocess.Popen(command,
                                      cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
        keep_awake(True)

    def refresh_stack(self):
        output = tail(ROOT / 'stacks' / f'{self.stack.name}-console.log')
        if output != self.feed.get('1.0', 'end-1c'):
            self.set_text(self.feed, output)
            self.feed.see('end-1c')
        if not (self.stack / 'stack.json').exists():
            self.status.set('Stack not created yet — press Start building')
            return
        state = stack_state(self.stack)
        mine = self.running()
        outside = self.child is None and not mine and state['fresh']
        done = not self.options.continuous and state['frozen'] >= self.options.blocks
        failed = self.child is not None and self.child.poll() not in (None, 0) and not self.stopped
        if not mine:
            keep_awake(False)
        if failed:
            status = 'Training stopped with an error — see trainer output'
        elif mine and self.stopped:
            status = 'Stopping and saving checkpoint'
        elif mine:
            status = f'Building block {state["stage"] + 1}'
        elif outside:
            status = f'Block {state["stage"] + 1} is training in another window'
        elif self.options.train_only_steps and self.child is not None and self.child.poll() == 0 and not self.stopped:
            status = 'Candidate saved for evaluation'
        elif done:
            status = f'Stack complete — {state["frozen"]} blocks saved'
        else:
            status = 'Paused — press Continue building'
        self.status.set(status)
        self.continue_button.configure(text='Continue building' if state['total_steps'] else 'Start building',
                                       state='disabled' if (mine or outside or done) else 'normal')
        self.stop_button.configure(state='normal' if mine and not self.stopped else 'disabled')
        if self.options.continuous:
            self.progress.configure(maximum=2000, value=min(state['stage_step'], 2000))
        else:
            self.progress.configure(maximum=self.options.blocks, value=state['frozen'])
        depth = str(state['frozen']) if self.options.continuous else f'{state["frozen"]} of {self.options.blocks}'
        parts = [f'{depth} blocks frozen ({state["parameters"]:,} parameters, width {state["width"]})',
                 f'block {state["stage"] + 1} at update {state["stage_step"]:,}']
        if state['validation'] is not None:
            parts.append(f'validation {state["validation"]:.4f} bits/byte')
        if state['seconds_per_step']:
            parts.append(f'{state["seconds_per_step"]:.2f} s per update')
        self.detail.set('  |  '.join(parts))
        self.draw_chart(state['points'], state['boundaries'], skip_first=True)
        review = state['crawl_review']
        if state['is_crawl']:
            if review is None:
                self.sample_title.set('Raw crawl review: no samples saved yet')
            elif review['key'] != self.last_sample:
                text = '\n\n'.join(
                    (sample.get("label", "") + "\n" if "label" in sample else "") + f'Prompt: {sample["prompt"]}\nSeed: {sample["seed"]}\nRaw continuation:\n{sample["response"]}'
                    for sample in review['samples']
                )
                self.set_text(self.sample, text)
                self.sample.see('end-1c')
                block, step = review['stage'] + 1, review['step']
                self.sample_title.set(f'Raw crawl review: block {block}, update {step:,} ({len(review["samples"])} prompts)')
                self.last_sample = review['key']
        elif state['samples'] and state['sample_at'] != self.last_sample:
            self.set_text(self.sample, '\n\n'.join(item['completion'] for item in state['samples']))
            self.sample_title.set(f'Fixed-prompt samples (block {state["sample_at"][0] + 1}, update {state["sample_at"][1]:,})')
            self.last_sample = state['sample_at']
        self.saved.set(f'Stack folder: {self.stack}   •   closing this window stops training; it resumes from the last saved update')

    # ------------------------------------------------------------------ baseline (unchanged behaviour)

    def continue_run(self):
        if not self.folder or self.running():
            return
        metrics = rows(self.folder / 'metrics.jsonl')
        self.target = (metrics[-1]['step'] if metrics else 0) + 1000
        self.log = (self.folder / 'console.log').open('a', encoding='utf-8')
        self.child = subprocess.Popen([str(PYTHON), '-u', str(ROOT / 'baseline.py'),
                                       '--resume', str(self.folder), '--steps', '1000'],
                                      cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
        self.continue_button.configure(state='disabled')

    def refresh_baseline(self):
        folders = sorted((ROOT / 'runs').glob('tinystories-*'))
        if not folders:
            self.status.set('No TinyStories run yet')
            self.continue_button.configure(state='disabled')
            return
        self.folder = folders[-1]
        evaluations = rows(self.folder / 'evaluations.jsonl')
        metrics = rows(self.folder / 'metrics.jsonl')
        result_path = self.folder / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        active = self.running() or result is None
        step = max([r['step'] for r in evaluations + metrics] + [0])
        failed = self.child is not None and self.child.poll() not in (None, 0) and not self.stopped
        self.status.set('Training failed — see console.log' if failed else
                        ('Training in progress' if active else 'Run complete — model saved'))
        self.continue_button.configure(state='disabled' if active else 'normal')
        self.progress.configure(maximum=max(self.target, step), value=step)
        loss = evaluations[-1]['validation_bpb'] if evaluations else None
        self.detail.set(f'Update {step:,} / {max(self.target, step):,}  |  Validation: ' +
                        (f'{loss:.4f} bits/byte' if loss is not None else 'pending'))
        self.draw_chart(evaluations)
        sample_key = (self.folder, evaluations[-1]['step']) if evaluations else None
        if evaluations and sample_key != self.last_sample:
            self.set_text(self.sample, '\n\n'.join(item['completion'] for item in evaluations[-1]['samples']))
            self.last_sample = sample_key
        self.saved.set(f'Checkpoint folder: {self.folder}\nlatest.pt = newest state • best.pt = lowest measured validation loss')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', help='stack name under stacks/ (default: the most recently used real stack, if any)')
    parser.add_argument('--baseline', action='store_true', help='show the TinyStories baseline run instead of a stack')
    parser.add_argument('--start', action='store_true', help='start building as soon as the window opens')
    parser.add_argument('--blocks', type=int, default=12)
    parser.add_argument('--train-only-steps', type=int, default=0)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--threads', type=int, default=1, help='CPU threads for training')
    parser.add_argument('--no-cache', action='store_true', help='run frozen blocks live instead of building the activation cache')
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--train-bytes', type=int, default=8388608)
    parser.add_argument('--check', action='store_true', help='print what the window would show and exit')
    options = parser.parse_args()
    if options.baseline:
        options.stack = None
    elif options.stack:
        options.stack = ROOT / 'stacks' / options.stack
    else:
        found = [p.parent for p in (ROOT / 'stacks').glob('*/stack.json') if not p.parent.name.startswith('smoke')]
        options.stack = max(found, key=lambda p: (p / 'stack.json').stat().st_mtime) if found else None
    if options.check:
        state = stack_state(options.stack) if options.stack and (options.stack / 'stack.json').exists() else None
        if state:
            state['points'] = state['points'][-3:]
            state['samples'] = bool(state['samples'])
        print(json.dumps({'stack': str(options.stack), 'state': state}, indent=1))
        return
    Monitor(options)


if __name__ == '__main__':
    main()
