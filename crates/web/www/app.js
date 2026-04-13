import init, { WaferRepl } from './pkg/wafer_web.js';

let repl = null;
const history = [];
let historyIdx = -1;

const WORD_CATEGORIES = {
  'Stack': 'DUP DROP SWAP OVER ROT NIP TUCK 2DUP 2DROP 2SWAP 2OVER PICK ROLL DEPTH .S'.split(' '),
  'Arithmetic': '+ - * / MOD /MOD NEGATE ABS MIN MAX M* UM* UM/MOD FM/MOD SM/REM */ */MOD'.split(' '),
  'Comparison': '= <> < > 0= 0< 0<> 0> U< U> WITHIN'.split(' '),
  'Logic': 'AND OR XOR INVERT LSHIFT RSHIFT TRUE FALSE'.split(' '),
  'Memory': '@ ! C@ C! +! HERE ALLOT , C, CELLS CELL+ MOVE FILL ERASE BLANK'.split(' '),
  'I/O': '. U. .R U.R EMIT CR SPACE SPACES TYPE ." .( .S'.split(' '),
  'Defining': ': ; VARIABLE CONSTANT VALUE CREATE DOES> DEFER IS TO :NONAME IMMEDIATE'.split(' '),
  'Control': 'IF ELSE THEN DO LOOP +LOOP I J LEAVE BEGIN UNTIL WHILE REPEAT AGAIN ?DO CASE OF ENDOF ENDCASE EXIT RECURSE'.split(' '),
  'Strings': 'S" S\\" C" COUNT COMPARE SEARCH /STRING -TRAILING'.split(' '),
  'Double': 'S>D D>S D+ D- DNEGATE DABS D= D< D. D.R 2@ 2! 2CONSTANT 2VARIABLE'.split(' '),
};

const output = document.getElementById('output');
const input = document.getElementById('input');
const prompt = document.getElementById('prompt');
const stackBar = document.getElementById('stack-bar');

function appendLine(text, cls) {
  const span = document.createElement('span');
  span.className = `line ${cls}`;
  span.textContent = text + '\n';
  output.appendChild(span);
  output.scrollTop = output.scrollHeight;
}

function updatePrompt() {
  if (!repl) return;
  prompt.textContent = repl.is_compiling() ? '] ' : '> ';
}

function updateStack() {
  if (!repl) return;
  try {
    const stack = repl.data_stack();
    if (stack.length === 0) {
      stackBar.textContent = 'Stack: (empty)';
    } else {
      stackBar.textContent = `Stack <${stack.length}> ${stack.join(' ')}`;
    }
  } catch {
    stackBar.textContent = 'Stack: (error)';
  }
}

function updateUserWords() {
  const cat = document.getElementById('cat-user');
  if (!cat) return;
  // We'll track user words by checking what the REPL evaluates
  // For now, just show the category
}

function evaluate(line) {
  if (!repl) return;
  const trimmed = line.trim();
  if (!trimmed) return;

  // Add to history
  history.push(trimmed);
  historyIdx = history.length;

  try {
    const result = repl.evaluate(trimmed);
    // Show input + output on one line (traditional Forth style)
    const combined = result.length > 0 ? `${trimmed} ${result} ok` : `${trimmed}  ok`;
    appendLine(combined, 'line-ok');
  } catch (e) {
    const msg = e.message || String(e);
    appendLine(`${trimmed}`, 'line-input');
    appendLine(`Error: ${msg}`, 'line-error');
  }

  updatePrompt();
  updateStack();
}

// Input handling
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    evaluate(input.value);
    input.value = '';
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    if (historyIdx > 0) {
      historyIdx--;
      input.value = history[historyIdx];
    }
  } else if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (historyIdx < history.length - 1) {
      historyIdx++;
      input.value = history[historyIdx];
    } else {
      historyIdx = history.length;
      input.value = '';
    }
  }
});

// Click output to focus input
output.addEventListener('click', () => input.focus());

// Word panel
document.getElementById('btn-toggle-words').addEventListener('click', () => {
  document.getElementById('word-panel').classList.toggle('collapsed');
});

function buildWordPanel() {
  const container = document.getElementById('word-categories');
  container.innerHTML = '';
  for (const [name, words] of Object.entries(WORD_CATEGORIES)) {
    const cat = document.createElement('div');
    cat.className = 'word-category';
    const h4 = document.createElement('h4');
    h4.textContent = name;
    h4.addEventListener('click', () => cat.classList.toggle('collapsed'));
    cat.appendChild(h4);
    const list = document.createElement('div');
    list.className = 'word-list';
    for (const w of words) {
      const chip = document.createElement('span');
      chip.className = 'word-chip';
      chip.textContent = w;
      chip.title = w;
      chip.addEventListener('click', () => {
        input.value += (input.value.length > 0 ? ' ' : '') + w;
        input.focus();
      });
      list.appendChild(chip);
    }
    cat.appendChild(list);
    container.appendChild(cat);
  }

  // User words category (dynamic)
  const userCat = document.createElement('div');
  userCat.className = 'word-category';
  userCat.id = 'cat-user';
  const h4 = document.createElement('h4');
  h4.textContent = 'User Words';
  h4.addEventListener('click', () => userCat.classList.toggle('collapsed'));
  userCat.appendChild(h4);
  const userList = document.createElement('div');
  userList.className = 'word-list';
  userList.id = 'user-word-list';
  userCat.appendChild(userList);
  container.appendChild(userCat);
}

// Word filter
document.getElementById('word-filter').addEventListener('input', (e) => {
  const q = e.target.value.toUpperCase();
  document.querySelectorAll('.word-chip').forEach(chip => {
    chip.style.display = chip.textContent.toUpperCase().includes(q) ? '' : 'none';
  });
});

// Init code panel
document.getElementById('btn-toggle-init').addEventListener('click', () => {
  const body = document.querySelector('.init-body');
  body.classList.toggle('collapsed');
  const btn = document.getElementById('btn-toggle-init');
  btn.innerHTML = body.classList.contains('collapsed') ? '&#x25BC;' : '&#x25B2;';
});

document.getElementById('btn-run-init').addEventListener('click', () => {
  const code = document.getElementById('init-code').value;
  if (code.trim()) {
    // Run each line separately
    for (const line of code.split('\n')) {
      if (line.trim()) evaluate(line);
    }
  }
  localStorage.setItem('wafer-init-code', code);
});

// Save init code on change
document.getElementById('init-code').addEventListener('input', (e) => {
  localStorage.setItem('wafer-init-code', e.target.value);
});

// Help
document.getElementById('btn-help').addEventListener('click', () => {
  document.getElementById('help-overlay').classList.remove('hidden');
});

document.getElementById('help-overlay').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) {
    document.getElementById('help-overlay').classList.add('hidden');
  }
});

document.querySelector('.close-help').addEventListener('click', () => {
  document.getElementById('help-overlay').classList.add('hidden');
});

// Reset
document.getElementById('btn-reset').addEventListener('click', () => {
  if (!repl) return;
  try {
    repl.reset();
    output.innerHTML = '';
    appendLine('WAFER reset.', 'line-ok');
    updatePrompt();
    updateStack();
  } catch (e) {
    appendLine(`Reset error: ${e.message}`, 'line-error');
  }
});

// Boot
async function boot() {
  output.innerHTML = '<div class="loading"><span>Loading WAFER...</span></div>';
  try {
    await init();
    repl = new WaferRepl();
    output.innerHTML = '';
    appendLine('WAFER — WebAssembly Forth Engine in Rust', 'line-output');
    appendLine(`Type Forth at the > prompt. Press ? for help.`, 'line-output');
    appendLine('', 'line-output');
    updatePrompt();
    updateStack();
    buildWordPanel();

    // Restore and run init code
    const saved = localStorage.getItem('wafer-init-code');
    if (saved) {
      document.getElementById('init-code').value = saved;
      for (const line of saved.split('\n')) {
        if (line.trim()) evaluate(line);
      }
    }

    // Load from URL hash if present
    if (location.hash.length > 1) {
      try {
        const code = atob(location.hash.slice(1));
        document.getElementById('init-code').value = code;
        for (const line of code.split('\n')) {
          if (line.trim()) evaluate(line);
        }
      } catch { /* ignore bad hash */ }
    }

    input.focus();
  } catch (e) {
    output.innerHTML = '';
    appendLine(`Failed to initialize WAFER: ${e.message || e}`, 'line-error');
    console.error(e);
  }
}

boot();
