const form = document.getElementById('customerForm');
const resultBox = document.getElementById('resultBox');
const historyList = document.getElementById('historyList');

async function loadHistory() {
  try {
    const response = await fetch('/api/history');
    const data = await response.json();

    historyList.innerHTML = '';

    if (!Array.isArray(data) || data.length === 0) {
      historyList.innerHTML = '<li>No records yet.</li>';
      return;
    }

    data.slice(0, 5).forEach((item) => {
      const li = document.createElement('li');
      li.textContent = `${item.created_at} - ${item.prediction} (${item.probability})`;
      historyList.appendChild(li);
    });
  } catch (error) {
    console.error('History error:', error);
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const result = await response.json();

    if (!response.ok) {
      resultBox.className = 'result-box neutral';
      resultBox.innerHTML = `<p class="result-text">Error: ${result.error || 'Prediction failed'}</p>`;
      return;
    }

    const predictionText = result.prediction === 'Yes' ? 'Will Churn' : 'Will Not Churn';
    resultBox.className = result.prediction === 'Yes' ? 'result-box churn' : 'result-box no-churn';
    resultBox.innerHTML = `
      <div>
        <p class="result-text">${predictionText}</p>
        <p>Confidence: ${result.confidence}%</p>
      </div>
    `;

    await loadHistory();
  } catch (error) {
    resultBox.className = 'result-box neutral';
    resultBox.innerHTML = `<p class="result-text">Connection error</p>`;
    console.error(error);
  }
});

loadHistory();
