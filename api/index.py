from flask import Flask, request, jsonify
import csv
import random
import math
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import statistics
import os

app = Flask(__name__)

# Constants
WHITE_BALL_MAX = 69
POWERBALL_MAX = 26
JACKPOT_ODDS = math.comb(69, 5) * 26

# Data structures
class PowerballDraw:
    def __init__(self, date, white_balls, powerball, multiplier=1):
        self.date = date
        self.white_balls = tuple(sorted(white_balls))
        self.powerball = powerball
        self.multiplier = multiplier

# Load data
def load_data():
    draws = []
    data_path = os.path.join(os.path.dirname(__file__), '..', 'historical_data.csv')

    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), 'historical_data.csv')

    if not os.path.exists(data_path):
        # Try absolute path for Vercel
        data_path = '/var/task/historical_data.csv'

    try:
        with open(data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    date = datetime.strptime(row['Draw Date'], '%m/%d/%Y')
                    numbers = row['Winning Numbers'].split()
                    if len(numbers) >= 6:
                        white = tuple(int(n) for n in numbers[:5])
                        pb = int(numbers[5])
                        mult = int(row.get('Multiplier', 1) or 1)
                        draws.append(PowerballDraw(date, white, pb, mult))
                except:
                    continue
        draws.sort(key=lambda x: x.date)
    except:
        pass
    return draws

draws = load_data()

# Frequency analysis
def get_frequencies():
    white_freq = Counter()
    pb_freq = Counter()
    for draw in draws:
        for num in draw.white_balls:
            white_freq[num] += 1
        pb_freq[draw.powerball] += 1
    return white_freq, pb_freq

white_freq, pb_freq = get_frequencies()

# Check historical match
def check_historical_match(white_balls, powerball):
    white_set = set(white_balls)
    for draw in draws:
        draw_white_set = set(draw.white_balls)
        white_match = len(white_set & draw_white_set)
        pb_match = powerball == draw.powerball

        if white_match == 5 and pb_match:
            return {'match_type': 'JACKPOT', 'prize': 'JACKPOT', 'date': draw.date.strftime('%m/%d/%Y'), 'white_matches': 5, 'powerball_match': True}
        elif white_match == 5:
            return {'match_type': '5+0', 'prize': '$1,000,000', 'date': draw.date.strftime('%m/%d/%Y'), 'white_matches': 5, 'powerball_match': False}
        elif white_match == 4 and pb_match:
            return {'match_type': '4+PB', 'prize': '$50,000', 'date': draw.date.strftime('%m/%d/%Y'), 'white_matches': 4, 'powerball_match': True}
    return None

# Generate tickets
def generate_ticket(strategy='random'):
    if strategy == 'random':
        white = tuple(sorted(random.sample(range(1, 70), 5)))
        pb = random.randint(1, 26)
    elif strategy in ['frequency', 'importance', 'importance_hot']:
        weights = [white_freq.get(n, 1) for n in range(1, 70)]
        total = sum(weights)
        probs = [w/total for w in weights]
        white = []
        available = list(range(1, 70))
        avail_probs = probs.copy()
        for _ in range(5):
            p_sum = sum(avail_probs)
            norm = [p/p_sum for p in avail_probs]
            choice = random.choices(available, weights=norm, k=1)[0]
            white.append(choice)
            idx = available.index(choice)
            available.pop(idx)
            avail_probs.pop(idx)
        pb_weights = [pb_freq.get(i, 1) for i in range(1, 27)]
        pb = random.choices(range(1, 27), weights=pb_weights, k=1)[0]
        white = tuple(sorted(white))
    elif strategy == 'hybrid':
        hot = [n for n, _ in Counter(white_freq).most_common(15)]
        white = set(random.sample(hot[:10], 2))
        while len(white) < 5:
            white.add(random.randint(1, 69))
        white = tuple(sorted(white))
        pb = random.randint(1, 26)
    else:
        white = tuple(sorted(random.sample(range(1, 70), 5)))
        pb = random.randint(1, 26)

    return white, pb, strategy

@app.route('/api/generate', methods=['POST', 'OPTIONS'])
def api_generate():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json or {}
    count = min(int(data.get('count', 1)), 100)
    strategy = data.get('strategy', 'mixed')

    strategies = ['random', 'frequency', 'hybrid', 'importance']
    tickets = []

    for i in range(count):
        strat = strategies[i % len(strategies)] if strategy == 'mixed' else strategy
        white, pb, s = generate_ticket(strat)
        historical_match = check_historical_match(white, pb)
        tickets.append({
            'white_balls': list(white),
            'powerball': pb,
            'strategy': s,
            'historical_match': historical_match
        })

    return jsonify({'tickets': tickets, 'cost': count * 2, 'count': count})

@app.route('/api/analysis', methods=['GET'])
def api_analysis():
    expected_white = (len(draws) * 5) / 69
    expected_pb = len(draws) / 26

    hot_white = [(n, c, ((c - expected_white) / expected_white) * 100) for n, c in Counter(white_freq).most_common(15)]
    cold_white = [(n, c, ((c - expected_white) / expected_white) * 100) for n, c in Counter(white_freq).most_common()[:-16:-1]]

    return jsonify({
        'hot_white': [{'number': n, 'count': c, 'diff': d} for n, c, d in hot_white],
        'cold_white': [{'number': n, 'count': c, 'diff': d} for n, c, d in cold_white],
        'hot_pb': [{'number': n, 'count': c} for n, c in Counter(pb_freq).most_common(10)],
        'cold_pb': [{'number': n, 'count': c} for n, c in Counter(pb_freq).most_common()[:-11:-1]],
        'total_draws': len(draws),
        'expected_white': expected_white,
        'expected_pb': expected_pb,
    })

@app.route('/api/recent-draws', methods=['GET'])
def api_recent():
    count = min(int(request.args.get('count', 20)), 50)
    recent = draws[-count:][::-1]
    return jsonify([{
        'date': d.date.strftime('%m/%d/%Y'),
        'white_balls': list(d.white_balls),
        'powerball': d.powerball,
        'multiplier': d.multiplier
    } for d in recent])

@app.route('/api/info', methods=['GET'])
def api_info():
    return jsonify({
        'total_draws': len(draws),
        'jackpot_odds': JACKPOT_ODDS,
        'latest_draw': draws[-1].date.strftime('%m/%d/%Y') if draws else None
    })

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# For local testing
if __name__ == '__main__':
    app.run(debug=True, port=5050)
