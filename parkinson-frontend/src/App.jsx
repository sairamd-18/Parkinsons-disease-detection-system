import { useState } from "react";
import "./App.css";

function App() {

  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [prob, setProb] = useState(null);
  const [loading, setLoading] = useState(false);

  const uploadAudio = async () => {

    if (!file) {
      alert("Please select a WAV audio file");
      return;
    }

    setLoading(true);
    setResult("");
    setProb(null);

    const formData = new FormData();
    formData.append("audio", file);

    try {

      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      // Handle backend errors
      if (!response.ok) {
        setResult(data.error);
        setProb(null);
      } else {
        setResult(data.result);
        setProb(data.probability);
      }

    } catch (err) {
      setResult("Server connection error");
      setProb(null);
    }

    setLoading(false);
  };

  return (
    <div className="main">

      <div className="card">

        <h1 className="title">Parkinson Voice Detection</h1>

        <p className="subtitle">
          AI powered neurological voice analysis
        </p>

        <input
          type="file"
          accept=".wav"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <button onClick={uploadAudio}>
          Analyze Voice
        </button>

        {loading && <p className="loading">Analyzing audio...</p>}

        {result && (
          <div className="resultBox">
            <h2>{result}</h2>

            {prob !== null && (
              <p>
                Probability: {(prob * 100).toFixed(2)}%
              </p>
            )}
          </div>
        )}

      </div>

    </div>
  );
}

export default App;