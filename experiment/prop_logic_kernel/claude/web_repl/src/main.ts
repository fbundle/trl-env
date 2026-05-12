import { 
  S, 
  parseP, 
  parseT, 
  resolveState, 
  autoSolve, 
  toStringP, 
  toStringT 
} from 'ts-prop-logic-kernel';

class REPLManager {
  private stateHistory: S[] = [];
  private tacticHistory: string[] = [];

  constructor() {
    this.setupEventListeners();
  }

  private setupEventListeners() {
    document.getElementById('start-btn')?.addEventListener('click', () => this.startProof());
    document.getElementById('apply-btn')?.addEventListener('click', () => this.applyTactic());
    document.getElementById('auto-btn')?.addEventListener('click', () => this.runAutoSolve());
    document.getElementById('undo-btn')?.addEventListener('click', () => this.undo());
    document.getElementById('reset-btn')?.addEventListener('click', () => this.reset());
    
    document.getElementById('tactic-input')?.addEventListener('keypress', (e) => {
      if ((e as KeyboardEvent).key === 'Enter') this.applyTactic();
    });
    
    document.getElementById('new-prop-input')?.addEventListener('keypress', (e) => {
      if ((e as KeyboardEvent).key === 'Enter') this.startProof();
    });

    // Symbol Toolbar Logic
    document.querySelectorAll('.symbol-toolbar').forEach(toolbar => {
      const targetId = (toolbar as HTMLElement).dataset.target;
      if (!targetId) return;
      const targetInput = document.getElementById(targetId) as HTMLInputElement;
      
      toolbar.querySelectorAll('.sym-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const sym = btn.textContent || '';
          const start = targetInput.selectionStart || 0;
          const end = targetInput.selectionEnd || 0;
          const text = targetInput.value;
          targetInput.value = text.substring(0, start) + sym + text.substring(end);
          targetInput.focus();
          const newPos = start + sym.length;
          targetInput.setSelectionRange(newPos, newPos);
        });
      });
    });
  }

  private startProof() {
    const input = (document.getElementById('new-prop-input') as HTMLInputElement).value;
    if (!input) return;

    try {
      const prop = parseP(input);
      const initialState: S = {
        varCount: 0,
        sorryCount: 0,
        newCount: 0,
        stack: [{ hyp: new Map(), goal: prop }]
      };
      
      this.stateHistory = [initialState];
      this.tacticHistory = [];
      this.updateUI();
      
      document.getElementById('proof-container')?.classList.remove('hidden');
      document.getElementById('success-msg')?.classList.add('hidden');
    } catch (e) {
      alert(`Error parsing proposition: ${e}`);
    }
  }

  private applyTactic() {
    const input = (document.getElementById('tactic-input') as HTMLInputElement).value;
    if (!input || this.stateHistory.length === 0) return;

    try {
      const tactic = parseT(input);
      const currentState = this.stateHistory[this.stateHistory.length - 1];
      const newState = resolveState(tactic, true, currentState);

      if (newState) {
        this.stateHistory.push(newState);
        this.tacticHistory.push(input);
        this.updateUI();
        (document.getElementById('tactic-input') as HTMLInputElement).value = '';
      } else {
        alert('Tactic failed to apply to the current goal.');
      }
    } catch (e) {
      alert(`Error parsing tactic: ${e}`);
    }
  }

  private runAutoSolve() {
    if (this.stateHistory.length === 0) return;
    const currentState = this.stateHistory[this.stateHistory.length - 1];
    
    const result = autoSolve(10, currentState);
    if (result) {
      this.stateHistory.push(result.state);
      this.tacticHistory.push(`Auto: ${result.path.map(toStringT).join(', ')}`);
      this.updateUI();
    } else {
      alert('Auto-solver could not find a proof within depth 10.');
    }
  }

  private undo() {
    if (this.stateHistory.length > 1) {
      this.stateHistory.pop();
      this.tacticHistory.pop();
      this.updateUI();
    }
  }

  private reset() {
    this.stateHistory = [];
    this.tacticHistory = [];
    document.getElementById('proof-container')?.classList.add('hidden');
    document.getElementById('success-msg')?.classList.add('hidden');
  }

  private updateUI() {
    const currentState = this.stateHistory[this.stateHistory.length - 1];
    
    if (currentState.stack.length === 0) {
      document.getElementById('success-msg')?.classList.remove('hidden');
    }

    // Update Goal
    const goalDisplay = document.getElementById('goal-display');
    if (goalDisplay) {
      if (currentState.stack.length > 0) {
        goalDisplay.textContent = `⊢ ${toStringP(currentState.stack[0].goal)}`;
      } else {
        goalDisplay.textContent = 'None (Proof Complete)';
      }
    }

    // Update Hypotheses
    const hypList = document.getElementById('hyp-list');
    if (hypList) {
      hypList.innerHTML = '';
      if (currentState.stack.length > 0) {
        currentState.stack[0].hyp.forEach((prop, id) => {
          const div = document.createElement('div');
          div.className = 'hyp-item';
          div.textContent = `${id}: ${toStringP(prop)}`;
          hypList.appendChild(div);
        });
      }
    }

    // Update Stack Info
    const stackInfo = document.getElementById('stack-info');
    if (stackInfo) {
      stackInfo.textContent = `Remaining goals: ${currentState.stack.length}`;
    }

    // Update History
    const historyDiv = document.getElementById('history');
    if (historyDiv) {
      historyDiv.innerHTML = '';
      this.tacticHistory.forEach((t, i) => {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.textContent = `${i + 1}. > ${t}`;
        historyDiv.appendChild(div);
      });
      historyDiv.scrollTop = historyDiv.scrollHeight;
    }
  }
}

new REPLManager();
