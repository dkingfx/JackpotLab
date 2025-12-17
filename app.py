#!/usr/bin/env python3
"""
Powerball Analyzer - Web Application
Flask-based web interface for the Powerball Mathematical Analyzer
"""

from flask import Flask, render_template, request, jsonify
import os
import json
from powerball_analyzer import (
    load_historical_data, NumberGenerator, MatchSimulator,
    FrequencyAnalyzer, GapAnalyzer, PatternAnalyzer,
    PowerballDraw, JACKPOT_ODDS, WHITE_BALL_MAX, POWERBALL_MAX
)
from datetime import datetime
import random

app = Flask(__name__)

# Load data on startup
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, 'historical_data.csv')
draws = load_historical_data(data_file)
generator = NumberGenerator(draws)
simulator = MatchSimulator(draws)
freq_analyzer = FrequencyAnalyzer(draws)
gap_analyzer = GapAnalyzer(draws)
pattern_analyzer = PatternAnalyzer(draws)


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html',
                          total_draws=len(draws),
                          jackpot_odds=f"{JACKPOT_ODDS:,}",
                          latest_draw=draws[-1] if draws else None)


@app.route('/api/analysis')
def get_analysis():
    """Get analysis data."""
    patterns = pattern_analyzer.get_optimal_patterns()

    return jsonify({
        'hot_white': [{'number': n, 'count': c, 'diff': ((c - freq_analyzer.expected_frequency('white')) / freq_analyzer.expected_frequency('white')) * 100}
                      for n, c in freq_analyzer.get_hot_numbers(15)],
        'cold_white': [{'number': n, 'count': c, 'diff': ((c - freq_analyzer.expected_frequency('white')) / freq_analyzer.expected_frequency('white')) * 100}
                       for n, c in freq_analyzer.get_cold_numbers(15)],
        'hot_pb': [{'number': n, 'count': c} for n, c in freq_analyzer.get_hot_numbers(10, 'powerball')],
        'cold_pb': [{'number': n, 'count': c} for n, c in freq_analyzer.get_cold_numbers(10, 'powerball')],
        'overdue_white': [{'number': n, 'gap': g, 'ratio': r} for n, g, r in gap_analyzer.get_overdue_numbers(15)],
        'overdue_pb': [{'number': n, 'gap': g, 'ratio': r} for n, g, r in gap_analyzer.get_overdue_numbers(10, 'powerball')],
        'hot_pairs': [{'pair': list(p), 'count': c} for p, c in freq_analyzer.get_hot_pairs(10)],
        'patterns': {
            'even_odd': [{'evens': e, 'count': c, 'pct': c/len(draws)*100} for e, c in patterns['even_count']],
            'high_low': [{'highs': h, 'count': c, 'pct': c/len(draws)*100} for h, c in patterns['high_count']],
            'avg_sum': patterns['avg_sum'],
            'sum_range': patterns['sum_range'],
        },
        'total_draws': len(draws),
        'expected_white': freq_analyzer.expected_frequency('white'),
        'expected_pb': freq_analyzer.expected_frequency('powerball'),
    })


def check_historical_match(white_balls, powerball):
    """Check if ticket matches any historical winning draw."""
    white_set = set(white_balls)

    for draw in draws:
        draw_white_set = set(draw.white_balls)
        white_match = len(white_set & draw_white_set)
        pb_match = powerball == draw.powerball

        # Check for significant matches (3+ white or any with PB match)
        if white_match == 5 and pb_match:
            return {
                'match_type': 'JACKPOT',
                'prize': 'JACKPOT',
                'date': draw.date.strftime('%m/%d/%Y'),
                'white_matches': 5,
                'powerball_match': True
            }
        elif white_match == 5:
            return {
                'match_type': '5+0',
                'prize': '$1,000,000',
                'date': draw.date.strftime('%m/%d/%Y'),
                'white_matches': 5,
                'powerball_match': False
            }
        elif white_match == 4 and pb_match:
            return {
                'match_type': '4+PB',
                'prize': '$50,000',
                'date': draw.date.strftime('%m/%d/%Y'),
                'white_matches': 4,
                'powerball_match': True
            }

    return None


@app.route('/api/generate', methods=['POST'])
def generate_tickets():
    """Generate tickets."""
    data = request.json
    count = min(int(data.get('count', 1)), 1000)  # Max 1000 tickets
    strategy = data.get('strategy', 'mixed')

    tickets = generator.generate_tickets(count, strategy)

    # Check each ticket for historical matches
    ticket_data = []
    for t in tickets:
        historical_match = check_historical_match(t.white_balls, t.powerball)
        ticket_data.append({
            'white_balls': list(t.white_balls),
            'powerball': t.powerball,
            'strategy': t.strategy,
            'historical_match': historical_match
        })

    return jsonify({
        'tickets': ticket_data,
        'cost': count * 2,
        'count': count
    })


@app.route('/api/simulate', methods=['POST'])
def simulate_jackpot():
    """Run jackpot simulation."""
    data = request.json
    white_balls = tuple(sorted(data.get('white_balls', [])))
    powerball = data.get('powerball', 1)
    max_attempts = min(int(data.get('max_attempts', 1000000)), 10000000)

    from powerball_analyzer import PowerballTicket
    ticket = PowerballTicket(white_balls, powerball, 'user')

    result = simulator.simulate_until_jackpot(ticket, max_attempts)

    return jsonify({
        'hit_jackpot': result.matched_draw is not None,
        'attempts': result.attempts,
        'cost': result.attempts * 2,
        'partial_matches': result.partial_matches,
        'theoretical_odds': JACKPOT_ODDS,
        'vs_expected': result.attempts / JACKPOT_ODDS if result.matched_draw else None
    })


@app.route('/api/test-draw', methods=['POST'])
def test_against_draw():
    """Test tickets against a specific draw."""
    data = request.json
    tickets_data = data.get('tickets', [])
    draw_index = data.get('draw_index', -1)  # -1 for most recent

    target_draw = draws[draw_index]

    from powerball_analyzer import PowerballTicket
    tickets = [
        PowerballTicket(tuple(sorted(t['white_balls'])), t['powerball'], t.get('strategy', 'user'))
        for t in tickets_data
    ]

    results = simulator.evaluate_tickets(tickets, target_draw)

    return jsonify({
        'draw': {
            'date': target_draw.date.strftime('%m/%d/%Y'),
            'white_balls': list(target_draw.white_balls),
            'powerball': target_draw.powerball
        },
        'results': [
            {
                'ticket': {
                    'white_balls': list(t.white_balls),
                    'powerball': t.powerball
                },
                'white_matches': w,
                'powerball_match': pb,
                'prize': prize
            }
            for t, w, pb, prize in results
        ]
    })


@app.route('/api/recent-draws')
def recent_draws():
    """Get recent draws."""
    count = min(int(request.args.get('count', 20)), 100)
    recent = draws[-count:][::-1]

    return jsonify([
        {
            'date': d.date.strftime('%m/%d/%Y'),
            'white_balls': list(d.white_balls),
            'powerball': d.powerball,
            'multiplier': d.multiplier
        }
        for d in recent
    ])


@app.route('/api/frequency-chart')
def frequency_chart():
    """Get frequency data for visualization."""
    white_data = []
    for n in range(1, WHITE_BALL_MAX + 1):
        count = freq_analyzer.white_freq.get(n, 0)
        expected = freq_analyzer.expected_frequency('white')
        white_data.append({
            'number': n,
            'count': count,
            'expected': expected,
            'diff_pct': ((count - expected) / expected) * 100
        })

    pb_data = []
    for n in range(1, POWERBALL_MAX + 1):
        count = freq_analyzer.pb_freq.get(n, 0)
        expected = freq_analyzer.expected_frequency('powerball')
        pb_data.append({
            'number': n,
            'count': count,
            'expected': expected,
            'diff_pct': ((count - expected) / expected) * 100
        })

    return jsonify({
        'white_balls': white_data,
        'powerballs': pb_data
    })


if __name__ == '__main__':
    print(f"Loaded {len(draws)} historical draws")
    print(f"Starting Powerball Analyzer web server...")
    app.run(debug=True, port=5050)
