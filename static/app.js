const lastTrainedEl = document.getElementById("last-trained");
const lastUpdatedEl = document.getElementById("last-updated");
const jitterMeanEl = document.getElementById("jitter-mean");
const jitterStdEl = document.getElementById("jitter-std");
const headlineTitleEl = document.getElementById("headline-title");
const headlineDescriptionEl = document.getElementById("headline-description");
const headlineRelevantEl = document.getElementById("headline-relevant");
const headlineJitterEl = document.getElementById("headline-jitter");
const newHeadlineBtn = document.getElementById("new-headline-btn");

function formatTimestamp(isoString) {
  if (!isoString) return "—";
  return new Date(isoString).toLocaleString();
}

function formatNumber(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(4);
}

function parseHeadline(text) {
  const raw = (text || "").trim();
  if (!raw) return { title: "—", description: "—" };

  const splitIndex = raw.indexOf("|");
  if (splitIndex === -1) return { title: raw, description: "—" };

  const title = raw.slice(0, splitIndex).trim() || "—";
  const description = raw.slice(splitIndex + 1).trim() || "—";
  return { title, description };
}

async function fetchSummary() {
  try {
    const response = await fetch("/summary/");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    lastTrainedEl.textContent = formatTimestamp(data.status.last_trained);
    lastUpdatedEl.textContent = formatTimestamp(data.status.last_updated);
    jitterMeanEl.textContent = formatNumber(data.mean);
    jitterStdEl.textContent = `± ${formatNumber(data.std)}`;
  } catch (err) {
    console.error("Failed to fetch summary:", err);
  }
}

async function fetchHeadline() {
  try {
    const response = await fetch("/random/");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    const parsed = parseHeadline(data.headline);
    headlineTitleEl.textContent = parsed.title;
    headlineDescriptionEl.textContent = parsed.description;
    headlineRelevantEl.textContent = data.relevant ? "Yes" : "No";
    headlineJitterEl.textContent = data.relevant
      ? formatNumber(data.jitter)
      : "N/A";
  } catch (err) {
    console.error("Failed to fetch headline:", err);
  }
}

newHeadlineBtn.addEventListener("click", fetchHeadline);

fetchSummary();
fetchHeadline();
setInterval(fetchSummary, 60_000);
