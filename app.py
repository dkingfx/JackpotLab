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


# Prize structure (without Power Play)
PRIZE_TIERS = {
    (5, True): 'JACKPOT',
    (5, False): 1000000,
    (4, True): 50000,
    (4, False): 100,
    (3, True): 100,
    (3, False): 7,
    (2, True): 7,
    (1, True): 4,
    (0, True): 4,
}

def calculate_prize(white_matches, pb_match):
    """Calculate prize for a ticket based on matches."""
    key = (white_matches, pb_match)
    return PRIZE_TIERS.get(key, 0)

def check_ticket_against_draw(ticket_white, ticket_pb, draw):
    """Check a ticket against a specific draw and return match details."""
    white_set = set(ticket_white)
    draw_white_set = set(draw.white_balls)
    white_matches = len(white_set & draw_white_set)
    pb_match = ticket_pb == draw.powerball
    prize = calculate_prize(white_matches, pb_match)

    return {
        'white_matches': white_matches,
        'powerball_match': pb_match,
        'prize': prize,
        'matched_whites': list(white_set & draw_white_set),
    }

@app.route('/api/simulate', methods=['POST'])
def simulate_budget():
    """Simulate buying tickets and compare against a drawing."""
    data = request.json
    budget = min(int(data.get('budget', 10)), 10000)  # Max $10,000
    strategy = data.get('strategy', 'mixed')
    draw_index = data.get('draw_index', 0)  # 0 = most recent

    # Get target draw
    if not draws:
        return jsonify({'error': 'No draws available'}), 400

    target_draw = draws[-(draw_index + 1)] if draw_index < len(draws) else draws[-1]

    # Calculate tickets from budget ($2 per ticket)
    ticket_count = budget // 2
    if ticket_count == 0:
        return jsonify({'error': 'Budget too low (minimum $2)'}), 400

    # Generate tickets
    tickets = generator.generate_tickets(ticket_count, strategy)

    # Generate tickets and check against target draw
    results = {
        'tickets': [],
        'summary': {
            'total_spent': ticket_count * 2,
            'total_won': 0,
            'profit_loss': 0,
            'ticket_count': ticket_count,
            'winners': 0,
            'by_tier': {}
        },
        'target_draw': {
            'date': target_draw.date.strftime('%m/%d/%Y'),
            'white_balls': list(target_draw.white_balls),
            'powerball': target_draw.powerball,
            'multiplier': target_draw.multiplier
        }
    }

    tier_counts = {
        '5+PB': 0, '5+0': 0, '4+PB': 0, '4+0': 0,
        '3+PB': 0, '3+0': 0, '2+PB': 0, '1+PB': 0, '0+PB': 0, 'No Win': 0
    }

    for i, t in enumerate(tickets):
        match_result = check_ticket_against_draw(t.white_balls, t.powerball, target_draw)
        prize = match_result['prize']

        # Determine tier name
        wm = match_result['white_matches']
        pm = match_result['powerball_match']
        if prize == 'JACKPOT':
            tier = '5+PB'
            prize_value = 200000000  # Assume $200M jackpot
        elif prize > 0:
            tier = f"{wm}+{'PB' if pm else '0'}"
            prize_value = prize
        else:
            tier = 'No Win'
            prize_value = 0

        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        if prize_value > 0:
            results['summary']['total_won'] += prize_value
            results['summary']['winners'] += 1

        # Only include first 100 tickets in response to avoid huge payloads
        if i < 100:
            results['tickets'].append({
                'white_balls': list(t.white_balls),
                'powerball': t.powerball,
                'strategy': t.strategy,
                'white_matches': match_result['white_matches'],
                'powerball_match': match_result['powerball_match'],
                'matched_whites': match_result['matched_whites'],
                'prize': prize_value,
                'tier': tier
            })

    results['summary']['profit_loss'] = results['summary']['total_won'] - results['summary']['total_spent']
    results['summary']['by_tier'] = {k: v for k, v in tier_counts.items() if v > 0}
    results['summary']['roi'] = round((results['summary']['total_won'] / results['summary']['total_spent'] - 1) * 100, 2) if results['summary']['total_spent'] > 0 else 0

    return jsonify(results)


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
