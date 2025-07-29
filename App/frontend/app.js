// js/app.js
const API_BASE_URL = "http://localhost:5000";

function showLoading() {
  const el = document.getElementById("loading");
  if (el) el.classList.remove("hidden");
}
function hideLoading() {
  const el = document.getElementById("loading");
  if (el) el.classList.add("hidden");
}
function showNotification(message, type = "info") {
  const note = document.getElementById("notification");
  const span = document.getElementById("notification-message");
  if (note && span) {
    span.textContent = message;
    note.className = `notification ${type}`;
    note.classList.remove("hidden");
    setTimeout(hideNotification, 5000);
  }
}
function hideNotification() {
  const note = document.getElementById("notification");
  if (note) note.classList.add("hidden");
}

// Manual Form Submission
const manualForm = document.getElementById("manual-form");
if (manualForm) {
  manualForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = {
      patient_id: document.getElementById("patient-id").value || "Unknown",
      wbc_count: parseFloat(document.getElementById("wbc-count").value),
      platelet_count: parseFloat(document.getElementById("platelet-count").value),
      neutrophils: parseFloat(document.getElementById("neutrophils").value),
      lymphocytes: parseFloat(document.getElementById("lymphocytes").value),
      mpv: parseFloat(document.getElementById("mpv").value),
      pdw: parseFloat(document.getElementById("pdw").value),
      hemoglobin: parseFloat(document.getElementById("hemoglobin").value),
      hct: parseFloat(document.getElementById("hct").value)
    };
    try {
      showLoading();
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });
      const result = await response.json();
      hideLoading();
      if (response.ok) {
        displayResult(result);
        showNotification("Diagnosis berhasil!", "success");
      } else {
        showNotification(`Error: ${result.error}`, "error");
      }
    } catch (err) {
      hideLoading();
      showNotification("Terjadi kesalahan koneksi", "error");
    }
  });
}

function displayResult(result) {
  const resultSection = document.getElementById("result-section");
  const resultContent = document.getElementById("result-content");
  const riskClass = result.prediction === "Positive" ? "risk-positive" : "risk-negative";
  resultContent.innerHTML = `
    <div class="result-card ${riskClass}">
      <h4>Hasil: ${result.prediction}</h4>
      <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
      <p><strong>Risk Level:</strong> ${result.risk_level}</p>
      <div class="probabilities">
        <div class="prob-item"><span>Negative:</span><span>${(result.probabilities.negative * 100).toFixed(1)}%</span></div>
        <div class="prob-item"><span>Positive:</span><span>${(result.probabilities.positive * 100).toFixed(1)}%</span></div>
      </div>
    </div>
  `;
  resultSection.classList.remove("hidden");
}

// CSV Submission
const csvForm = document.getElementById("csv-form");
if (csvForm) {
  csvForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("csv-file");
    const file = fileInput.files[0];
    if (!file) return showNotification("Pilih file CSV terlebih dahulu", "error");
    const formData = new FormData();
    formData.append("file", file);
    try {
      showLoading();
      const response = await fetch(`${API_BASE_URL}/predict/batch`, {
        method: "POST",
        body: formData
      });
      const result = await response.json();
      hideLoading();
      if (response.ok) {
        sessionStorage.setItem("csvResults", JSON.stringify(result));
        window.location.href = "hasil-diagnosa.html";
      } else {
        showNotification(`Error: ${result.error}`, "error");
      }
    } catch (err) {
      hideLoading();
      showNotification("Terjadi kesalahan saat upload", "error");
    }
  });
}

function displayCsvResults(result) {
  const csvResults = document.getElementById("csv-results");
  const csvSummary = document.getElementById("csv-summary");
  const csvDetails = document.getElementById("csv-details");
  csvSummary.innerHTML = `
    <div class="summary-grid">
      <div class="summary-item"><h4>${result.summary.total_samples}</h4><p>Total Sampel</p></div>
      <div class="summary-item positive"><h4>${result.summary.positive_cases}</h4><p>Kasus Positif</p></div>
      <div class="summary-item negative"><h4>${result.summary.negative_cases}</h4><p>Kasus Negatif</p></div>
      <div class="summary-item"><h4>${result.summary.positive_rate.toFixed(1)}%</h4><p>Persentase Positif</p></div>
    </div>
  `;
  let detailsHtml = '<div class="results-table"><table><thead><tr><th>Baris</th><th>Prediksi</th><th>Confidence</th><th>Risk Level</th></tr></thead><tbody>';
  result.results.forEach((item) => {
    const rowClass = item.prediction === "Positive" ? "positive-row" : "negative-row";
    detailsHtml += `<tr class="${rowClass}"><td>${item.row}</td><td>${item.prediction}</td><td>${(item.confidence * 100).toFixed(1)}%</td><td>${item.risk_level}</td></tr>`;
  });
  detailsHtml += "</tbody></table></div>";
  csvDetails.innerHTML = detailsHtml;
  csvResults.classList.remove("hidden");
}

const historyContent = document.getElementById("history-content");
if (historyContent) loadHistory();

async function loadHistory() {
  try {
    showLoading();
    const response = await fetch(`${API_BASE_URL}/history`);
    const result = await response.json();
    hideLoading();
    if (response.ok) displayHistory(result.history);
    else showNotification(`Error: ${result.error}`, "error");
  } catch (err) {
    hideLoading();
    showNotification("Gagal memuat riwayat", "error");
  }
}

function displayHistory(history) {
  if (!history.length) return historyContent.innerHTML = "<p>Belum ada riwayat diagnosis.</p>";
  let html = '<div class="history-table"><table><thead><tr><th>Waktu</th><th>Patient ID</th><th>Prediksi</th><th>Confidence</th><th>Risk Level</th></tr></thead><tbody>';
  history.forEach((item) => {
    const date = new Date(item.timestamp).toLocaleString("id-ID", { timeZone: "Asia/Jakarta" });
    const rowClass = item.prediction === "Positive" ? "positive-row" : "negative-row";
    html += `<tr class="${rowClass}"><td>${date}</td><td>${item.patient_id}</td><td>${item.prediction}</td><td>${(item.confidence * 100).toFixed(1)}%</td><td>${item.risk_level}</td></tr>`;
  });
  html += "</tbody></table></div>";
  historyContent.innerHTML = html;
}

const statsModal = document.getElementById("stats-modal");
if (statsModal) {
  window.loadStatistics = async function () {
    try {
      showLoading();
      const response = await fetch(`${API_BASE_URL}/stats`);
      const result = await response.json();
      hideLoading();
      if (response.ok) {
        sessionStorage.setItem("csvResults", JSON.stringify(result));
        window.location.href = "hasil-diagnosa.html";
      } else {
        showNotification(`Error: ${result.error}`, "error");
      }
    } catch (err) {
      hideLoading();
      showNotification("Gagal memuat statistik", "error");
    }
  }

  window.closeModal = () => {
    statsModal.style.display = "none";
  }
}

function displayStatistics(stats) {
  const statsContent = document.getElementById("stats-content");
  let html = `<div class="stats-grid"><div class="stat-item"><h4>${stats.total_diagnoses}</h4><p>Total Diagnosis</p></div>`;
  if (stats.prediction_distribution) {
    Object.entries(stats.prediction_distribution).forEach(([k, v]) => {
      html += `<div class="stat-item"><h4>${v}</h4><p>${k}</p></div>`;
    });
  }
  html += "</div>";
  if (stats.daily_activity?.length) {
    html += '<h4>Aktivitas 7 Hari Terakhir</h4><div class="activity-chart">';
    stats.daily_activity.forEach((d) => {
      html += `<div class="activity-item"><span>${d.date}</span><span>${d.count} diagnosis</span></div>`;
    });
    html += '</div>';
  }
  statsContent.innerHTML = html;
}

const csvInput = document.getElementById("csv-file");
if (csvInput) {
  csvInput.addEventListener("change", function (e) {
    const fileName = e.target.files[0]?.name || "Pilih File CSV";
    const label = e.target.nextElementSibling.querySelector("span");
    if (label) label.textContent = fileName;
  });
}

// Health check
window.addEventListener("DOMContentLoaded", () => {
  fetch(`${API_BASE_URL}/health`)
    .then((res) => res.json())
    .then((data) => {
      if (data.status === "healthy" && data.model_loaded) showNotification("Aplikasi siap digunakan", "success");
      else showNotification("Model belum dimuat dengan benar", "warning");
    })
    .catch(() => showNotification("Tidak dapat terhubung ke server", "error"));

  const csvData = sessionStorage.getItem("csvResults");
  if (csvData) {
    const parsed = JSON.parse(csvData);
    displayCsvResults(parsed);
    sessionStorage.removeItem("csvResults");
  }
});

async function clearHistory() {
  if (!confirm("Yakin ingin menghapus semua riwayat diagnosis?")) return;
  try {
    showLoading();
    const response = await fetch(`${API_BASE_URL}/history/clear`, {
      method: "POST"
    });
    const result = await response.json();
    hideLoading();
    if (response.ok) {
      showNotification(result.message, "success");
      loadHistory(); // Refresh tampilan tabel riwayat
    } else {
      showNotification(`Error: ${result.error}`, "error");
    }
  } catch (err) {
    hideLoading();
    showNotification("Gagal menghapus riwayat", "error");
  }
}


