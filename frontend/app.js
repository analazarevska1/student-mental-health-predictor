// ── Config ─────────────────────────────────────────────
const API_URL = 'http://localhost:5001/predict';
;
const TOTAL_SECS  = 5;

// ── State ──────────────────────────────────────────────
let currentSection = 1;

// ── Screen helpers ─────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

function startSurvey() {
  showScreen('survey');
  showSection(1);
  // Re-init sliders after survey becomes visible (offsetWidth now available)
  setTimeout(() => {
    document.querySelectorAll('.slider').forEach(s => updateSlider(s.id, s.value));
  }, 50);
}

function retake() {
  currentSection = 1;

  // Reset number inputs
  ['age', 'study_hours_per_day', 'sleep_hours', 'social_q1'].forEach(id => {
    document.getElementById(id).value = '';
  });

  // Reset gender radio buttons
  document.querySelectorAll('input[name="gender"]').forEach(radio => {
    radio.checked = false;
  });

  // Reset sliders to their default values
  const sliderDefaults = {
    exam_pressure: 5, family_expectation: 5,
    stress_q1: 5, stress_q2: 5,
    anxiety_q1: 5, anxiety_q2: 5,
    depression_q1: 5, depression_q2: 5,
    activity_q1: 3, activity_q2: 5,
    financial_stress: 5,
    social_q2: 5, social_q3: 5,
  };

  Object.entries(sliderDefaults).forEach(([id, defaultVal]) => {
    const slider = document.getElementById(id);
    if (slider) {
      slider.value = defaultVal;
      updateSlider(id, defaultVal);   // also resets the floating tooltip
    }
  });

  showScreen('landing');
}
// ── Section navigation ─────────────────────────────────
function showSection(n) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.getElementById(`section-${n}`).classList.add('active');

  document.getElementById('progress-bar').style.width  = `${(n / TOTAL_SECS) * 100}%`;
  document.getElementById('progress-label').textContent = `${n} of ${TOTAL_SECS}`;
  document.getElementById('nav-row').style.display = n === TOTAL_SECS ? 'none' : 'flex';

  currentSection = n;
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Re-position slider tooltips after section becomes visible
  setTimeout(() => {
    document.querySelectorAll(`#section-${n} .slider`).forEach(s => updateSlider(s.id, s.value));
  }, 50);
}

function nextSection() {
  if (!validateSection(currentSection)) return;
  if (currentSection < TOTAL_SECS) showSection(currentSection + 1);
}

function goBack() {
  if (currentSection > 1) {
    showSection(currentSection - 1);
  } else {
    showScreen('landing');
  }
}

// ── Slider labels — tooltip follows thumb ──────────────
function updateSlider(id, rawValue) {
  const slider = document.getElementById(id);
  const el     = document.getElementById(`${id}-val`);
  if (!el || !slider) return;

  const value  = parseFloat(rawValue);
  el.textContent = Number.isInteger(value) ? value : value.toFixed(1);

  // Calculate thumb position as fraction of track
  const min    = parseFloat(slider.min);
  const max    = parseFloat(slider.max);
  const pct    = (value - min) / (max - min);
  const thumbW = 20;
  const trackW = slider.offsetWidth;
  if (trackW === 0) return; // not visible yet
  const offset = pct * (trackW - thumbW) + thumbW / 2;
  el.style.left = `${offset}px`;
}

// ── Validation ─────────────────────────────────────────
function validateSection(n) {
  if (n === 1) {
    const age    = document.getElementById('age').value;
    const gender = document.querySelector('input[name="gender"]:checked');
    if (!age || age < 16 || age > 35) {
      showToast('Please enter a valid age (16–35)');
      return false;
    }
    if (!gender) {
      showToast('Please select how you identify');
      return false;
    }
  }
  if (n === 2) {
    const study = document.getElementById('study_hours_per_day').value;
    if (study === '' || study < 0 || study > 16) {
      showToast('Please enter valid study hours (0–16)');
      return false;
    }
  }
  if (n === 4) {
    const sleep = document.getElementById('sleep_hours').value;
    if (sleep === '' || sleep < 0 || sleep > 12) {
      showToast('Please enter valid sleep hours (0–12)');
      return false;
    }
  }
  if (n === 5) {
    const social = document.getElementById('social_q1').value;
    if (social === '' || social < 0 || social > 10) {
      showToast('Please enter valid hours with friends (0–10)');
      return false;
    }
  }
  return true;
}

// ── Build payload ──────────────────────────────────────
function buildPayload() {
  const avg    = (...vals) => vals.reduce((a, b) => a + b, 0) / vals.length;
  const slider = id => parseFloat(document.getElementById(id).value);
  const num    = id => parseFloat(document.getElementById(id).value);

  const stress_level     = avg(slider('stress_q1'), slider('stress_q2'));
  const anxiety_score    = avg(slider('anxiety_q1'), slider('anxiety_q2'));
  const depression_score = avg(slider('depression_q1'), slider('depression_q2'));

  const activity_days10  = (slider('activity_q1') / 7) * 10;
  const sedentary        = slider('activity_q2');
  const physical_activity = avg(activity_days10, 10 - sedentary);

  const social_hours     = Math.min(num('social_q1'), 10);
  const family_contact   = slider('social_q2');
  const loneliness_inv   = 10 - slider('social_q3');
  const social_support   = avg(social_hours, family_contact, loneliness_inv);

  return {
    age:                  num('age'),
    gender:               parseFloat(document.querySelector('input[name="gender"]:checked').value),
    study_hours_per_day:  num('study_hours_per_day'),
    exam_pressure:        slider('exam_pressure'),
    stress_level:         parseFloat(stress_level.toFixed(2)),
    anxiety_score:        parseFloat(anxiety_score.toFixed(2)),
    depression_score:     parseFloat(depression_score.toFixed(2)),
    sleep_hours:          num('sleep_hours'),
    physical_activity:    parseFloat(physical_activity.toFixed(2)),
    social_support:       parseFloat(social_support.toFixed(2)),
    financial_stress:     slider('financial_stress'),
    family_expectation:   slider('family_expectation'),
  };
}

// ── Submit ─────────────────────────────────────────────
async function submitSurvey() {
  if (!validateSection(5)) return;

  const payload = buildPayload();

  document.getElementById('submit-text').textContent = 'Analysing...';
  document.getElementById('submit-spinner').classList.remove('hidden');

  try {
    const res  = await fetch(API_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();
    showResults(data, payload);
} catch (err) {
    console.error(err);
    showToast('Error: ' + err.message);
}
   finally {
    document.getElementById('submit-text').textContent = 'Get my results';
    document.getElementById('submit-spinner').classList.add('hidden');
  }
  // console.log("Sending payload:", payload);
}

// ── Render results ─────────────────────────────────────
function showResults({ burnout_level, ai_message,tips, top_factors, closing }, payload) {
  showScreen('results');

  const card  = document.getElementById('risk-card');
  const label = document.getElementById('risk-label');
  const level = document.getElementById('risk-level');
  const desc  = document.getElementById('risk-desc');

  const classMap = { Low: 'low', Medium: 'med', High: 'high' };
  card.className = 'risk-card ' + (classMap[burnout_level] || 'med');
  label.textContent = 'Your burnout risk';
  level.textContent = `${burnout_level} risk`;
  desc.textContent  = ai_message;   // ← AI message now goes here


  const tipsList = document.getElementById('tips-list');
  tipsList.innerHTML = tips.map(tip => `
    <li>
      <span class="tip-icon">
        <svg viewBox="0 0 11 11" fill="none">
          <path d="M2 5.5l2.5 2.5 4.5-5" stroke="#4a7c59" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </span>
      ${tip}
    </li>
  `).join('');
  document.getElementById('closing-card').textContent = closing;


  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Toast ──────────────────────────────────────────────
let toastTimer;
function showToast(msg) {
  let toast = document.getElementById('toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast';
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3000);
}

// ── Init ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  showScreen('landing');
});

// Re-position tooltips on window resize
window.addEventListener('resize', () => {
  document.querySelectorAll('.slider').forEach(s => updateSlider(s.id, s.value));
});