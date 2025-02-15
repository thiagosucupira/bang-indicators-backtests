from flask import Flask, request, jsonify
import pandas as pd
from markov import run_markov_strategy_detailed

app = Flask(__name__)

@app.route('/api/markov-strategy', methods=['POST'])
def markov_strategy():
    # Expect JSON data representing market data (array of objects, each object is a row)
    data = request.get_json(force=True)
    # Convert JSON data to a DataFrame
    df = pd.DataFrame(data)
    result = run_markov_strategy_detailed(df)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 