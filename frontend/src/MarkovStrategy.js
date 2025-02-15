import React, { useState } from 'react';

function MarkovStrategy() {
  const [jsonInput, setJsonInput] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleRunStrategy = async () => {
    setError('');
    setLoading(true);
    try {
      const dataObj = JSON.parse(jsonInput);
      const response = await fetch('/api/markov-strategy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(dataObj)
      });
      if (!response.ok) {
        throw new Error('API error: ' + response.statusText);
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.toString());
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', margin: '20px' }}>
      <h2>Run Markov Strategy</h2>
      <textarea
        value={jsonInput}
        onChange={(e) => setJsonInput(e.target.value)}
        placeholder='Enter JSON market data (array of objects where each object is a row)'
        rows={10}
        cols={50}
      />
      <br />
      <button onClick={handleRunStrategy} disabled={loading} style={{ marginTop: '10px' }}>
        {loading ? 'Running...' : 'Run Strategy'}
      </button>
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {result && (
        <div style={{ marginTop: '20px' }}>
          <h3>Metrics:</h3>
          <pre>{JSON.stringify(result.metrics, null, 2)}</pre>
          <h3>Trades:</h3>
          <pre>{JSON.stringify(result.trades, null, 2)}</pre>
          {result.plotImage && (
            <div>
              <h3>Plot:</h3>
              <img
                src={`data:image/png;base64,${result.plotImage}`}
                alt='Markov Strategy Plot'
                style={{ maxWidth: '100%', height: 'auto' }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default MarkovStrategy; 