#!/usr/bin/env python3
"""
Powerball Mathematical Analyzer & Generator
============================================
A mathematically-informed lottery number generator that uses historical data
analysis to generate numbers using various probabilistic strategies.

Mathematical Approaches:
1. Frequency Analysis - Numbers that appear more/less often
2. Gap Analysis - Time since last appearance (overdue numbers)
3. Hot/Cold Streaks - Recent performance patterns
4. Pair/Triple Analysis - Numbers that appear together
5. Positional Analysis - Which numbers appear in which positions
6. Distribution Analysis - Even/odd, high/low ratios

Author: Mathematical Powerball Analyzer
"""

import csv
import random
import math
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
import statistics


# Powerball Constants
WHITE_BALL_MIN = 1
WHITE_BALL_MAX = 69
POWERBALL_MIN = 1
POWERBALL_MAX = 26
WHITE_BALL_COUNT = 5

# Odds calculation
TOTAL_WHITE_COMBINATIONS = math.comb(69, 5)  # 11,238,513
TOTAL_POWERBALL = 26
JACKPOT_ODDS = TOTAL_WHITE_COMBINATIONS * TOTAL_POWERBALL  # 292,201,338


@dataclass
class PowerballDraw:
    """Represents a single Powerball drawing."""
    date: datetime
    white_balls: Tuple[int, ...]
    powerball: int
    multiplier: int = 1

    def __post_init__(self):
        self.white_balls = tuple(sorted(self.white_balls))

    def matches(self, ticket: 'PowerballTicket') -> Tuple[int, bool]:
        """Returns (white_matches, powerball_match)"""
        white_matches = len(set(self.white_balls) & set(ticket.white_balls))
        pb_match = self.powerball == ticket.powerball
        return white_matches, pb_match

    def __str__(self):
        return f"{self.date.strftime('%m/%d/%Y')}: {' '.join(map(str, self.white_balls))} PB:{self.powerball}"


@dataclass
class PowerballTicket:
    """Represents a lottery ticket."""
    white_balls: Tuple[int, ...]
    powerball: int
    strategy: str = "random"

    def __post_init__(self):
        self.white_balls = tuple(sorted(self.white_balls))

    def __str__(self):
        return f"{' '.join(map(str, self.white_balls))} PB:{self.powerball} [{self.strategy}]"

    def __hash__(self):
        return hash((self.white_balls, self.powerball))

    def __eq__(self, other):
        return self.white_balls == other.white_balls and self.powerball == other.powerball


@dataclass
class MatchResult:
    """Result of matching a ticket against draws."""
    ticket: PowerballTicket
    attempts: int
    matched_draw: Optional[PowerballDraw] = None
    prize_tier: str = ""
    partial_matches: Dict[str, int] = field(default_factory=dict)


class FrequencyAnalyzer:
    """Analyzes frequency patterns in historical data."""

    def __init__(self, draws: List[PowerballDraw]):
        self.draws = draws
        self.white_freq = Counter()
        self.pb_freq = Counter()
        self.pair_freq = Counter()
        self.position_freq = {i: Counter() for i in range(5)}
        self.recent_window = 30  # Last N draws for hot/cold

        self._analyze()

    def _analyze(self):
        """Perform comprehensive frequency analysis."""
        for draw in self.draws:
            # Individual number frequency
            for num in draw.white_balls:
                self.white_freq[num] += 1
            self.pb_freq[draw.powerball] += 1

            # Positional frequency (sorted positions)
            for pos, num in enumerate(draw.white_balls):
                self.position_freq[pos][num] += 1

            # Pair frequency
            for i, n1 in enumerate(draw.white_balls):
                for n2 in draw.white_balls[i+1:]:
                    self.pair_freq[(min(n1, n2), max(n1, n2))] += 1

    def get_hot_numbers(self, n: int = 10, ball_type: str = 'white') -> List[Tuple[int, int]]:
        """Get most frequently drawn numbers."""
        freq = self.white_freq if ball_type == 'white' else self.pb_freq
        return freq.most_common(n)

    def get_cold_numbers(self, n: int = 10, ball_type: str = 'white') -> List[Tuple[int, int]]:
        """Get least frequently drawn numbers."""
        freq = self.white_freq if ball_type == 'white' else self.pb_freq
        max_num = WHITE_BALL_MAX if ball_type == 'white' else POWERBALL_MAX

        all_nums = {i: freq.get(i, 0) for i in range(1, max_num + 1)}
        return sorted(all_nums.items(), key=lambda x: x[1])[:n]

    def get_recent_hot(self, n: int = 10) -> List[Tuple[int, int]]:
        """Get hot numbers from recent draws."""
        recent = Counter()
        for draw in self.draws[-self.recent_window:]:
            for num in draw.white_balls:
                recent[num] += 1
        return recent.most_common(n)

    def get_hot_pairs(self, n: int = 10) -> List[Tuple[Tuple[int, int], int]]:
        """Get most common number pairs."""
        return self.pair_freq.most_common(n)

    def expected_frequency(self, ball_type: str = 'white') -> float:
        """Calculate expected frequency per number if perfectly random."""
        total_draws = len(self.draws)
        if ball_type == 'white':
            # Each draw picks 5 from 69, expected per number
            return (total_draws * 5) / 69
        else:
            return total_draws / 26


class GapAnalyzer:
    """Analyzes gaps between number appearances."""

    def __init__(self, draws: List[PowerballDraw]):
        self.draws = draws
        self.last_seen_white = {}
        self.last_seen_pb = {}
        self.avg_gaps_white = defaultdict(list)
        self.avg_gaps_pb = defaultdict(list)

        self._analyze()

    def _analyze(self):
        """Calculate gaps for all numbers."""
        white_last = {i: -1 for i in range(1, WHITE_BALL_MAX + 1)}
        # Handle historical Powerball ranges (was up to 35 before Oct 2015)
        max_pb_seen = max(d.powerball for d in self.draws)
        pb_last = {i: -1 for i in range(1, max(POWERBALL_MAX, max_pb_seen) + 1)}

        for idx, draw in enumerate(self.draws):
            # White balls
            for num in draw.white_balls:
                if white_last[num] >= 0:
                    gap = idx - white_last[num]
                    self.avg_gaps_white[num].append(gap)
                white_last[num] = idx

            # Powerball
            if pb_last[draw.powerball] >= 0:
                gap = idx - pb_last[draw.powerball]
                self.avg_gaps_pb[draw.powerball].append(gap)
            pb_last[draw.powerball] = idx

        # Store last seen index
        total = len(self.draws)
        self.last_seen_white = {n: total - 1 - v for n, v in white_last.items()}
        self.last_seen_pb = {n: total - 1 - v for n, v in pb_last.items()}

    def get_overdue_numbers(self, n: int = 10, ball_type: str = 'white') -> List[Tuple[int, int, float]]:
        """
        Get numbers that are "overdue" based on their average gap.
        Returns: (number, current_gap, gap_ratio) where gap_ratio > 1 means overdue
        """
        if ball_type == 'white':
            last_seen = self.last_seen_white
            avg_gaps = self.avg_gaps_white
        else:
            last_seen = self.last_seen_pb
            avg_gaps = self.avg_gaps_pb

        results = []
        for num, current_gap in last_seen.items():
            if avg_gaps[num]:
                avg = statistics.mean(avg_gaps[num])
                ratio = current_gap / avg if avg > 0 else 0
                results.append((num, current_gap, ratio))
            else:
                # Never drawn - very overdue
                results.append((num, current_gap, float('inf')))

        return sorted(results, key=lambda x: -x[2])[:n]

    def get_due_score(self, number: int, ball_type: str = 'white') -> float:
        """Calculate how 'due' a number is (higher = more overdue)."""
        if ball_type == 'white':
            current = self.last_seen_white.get(number, 0)
            gaps = self.avg_gaps_white.get(number, [])
        else:
            current = self.last_seen_pb.get(number, 0)
            gaps = self.avg_gaps_pb.get(number, [])

        if gaps:
            avg = statistics.mean(gaps)
            return current / avg if avg > 0 else 1.0
        return 1.0


class PatternAnalyzer:
    """Analyzes distribution patterns in draws."""

    def __init__(self, draws: List[PowerballDraw]):
        self.draws = draws
        self.even_odd_dist = []
        self.high_low_dist = []
        self.sum_dist = []
        self.range_dist = []

        self._analyze()

    def _analyze(self):
        """Analyze patterns in historical draws."""
        for draw in self.draws:
            nums = draw.white_balls

            # Even/Odd distribution
            evens = sum(1 for n in nums if n % 2 == 0)
            self.even_odd_dist.append(evens)

            # High/Low distribution (1-34 low, 35-69 high)
            highs = sum(1 for n in nums if n > 34)
            self.high_low_dist.append(highs)

            # Sum of numbers
            self.sum_dist.append(sum(nums))

            # Range (max - min)
            self.range_dist.append(max(nums) - min(nums))

    def get_optimal_patterns(self) -> Dict:
        """Get most common patterns from historical data."""
        return {
            'even_count': Counter(self.even_odd_dist).most_common(3),
            'high_count': Counter(self.high_low_dist).most_common(3),
            'avg_sum': statistics.mean(self.sum_dist),
            'sum_range': (min(self.sum_dist), max(self.sum_dist)),
            'avg_range': statistics.mean(self.range_dist),
        }

    def score_ticket(self, ticket: PowerballTicket) -> float:
        """Score a ticket based on how well it matches historical patterns."""
        nums = ticket.white_balls
        score = 0.0

        # Even/odd score
        evens = sum(1 for n in nums if n % 2 == 0)
        optimal_even = Counter(self.even_odd_dist).most_common(1)[0][0]
        score += 1.0 if evens == optimal_even else 0.5 if abs(evens - optimal_even) == 1 else 0

        # High/low score
        highs = sum(1 for n in nums if n > 34)
        optimal_high = Counter(self.high_low_dist).most_common(1)[0][0]
        score += 1.0 if highs == optimal_high else 0.5 if abs(highs - optimal_high) == 1 else 0

        # Sum score (within 1 std dev of mean)
        ticket_sum = sum(nums)
        mean_sum = statistics.mean(self.sum_dist)
        std_sum = statistics.stdev(self.sum_dist)
        if abs(ticket_sum - mean_sum) <= std_sum:
            score += 1.0
        elif abs(ticket_sum - mean_sum) <= 2 * std_sum:
            score += 0.5

        return score / 3.0  # Normalize to 0-1


class ImportanceSampler:
    """
    Importance Sampling Engine for Lottery Number Generation.

    Mathematical Foundation:
    =======================
    Standard Monte Carlo: P(n) = 1/N (uniform)
    Importance Sampling:  P(n) = w(n) / Σw(i) (weighted by observed data)

    This shifts the probability distribution toward historically observed
    patterns while still allowing all numbers to be selected.

    Key Insight: If we're trying to match historical draws, weighting toward
    observed frequencies will (on average) reach matches faster.
    """

    def __init__(self, freq_analyzer: FrequencyAnalyzer,
                 gap_analyzer: GapAnalyzer,
                 pattern_analyzer: PatternAnalyzer):
        self.freq = freq_analyzer
        self.gap = gap_analyzer
        self.pattern = pattern_analyzer

        # Precompute importance weights for all numbers
        self.white_weights = self._compute_white_weights()
        self.pb_weights = self._compute_pb_weights()

    def _compute_white_weights(self) -> Dict[int, float]:
        """
        Compute importance weights for white balls using multiple factors:
        - Frequency (how often it appears)
        - Recency (how recently it appeared)
        - Gap deviation (is it overdue?)
        """
        weights = {}

        for num in range(1, WHITE_BALL_MAX + 1):
            # Factor 1: Frequency weight (normalized)
            freq_count = self.freq.white_freq.get(num, 1)
            expected = self.freq.expected_frequency('white')
            freq_weight = freq_count / expected  # >1 means hot, <1 means cold

            # Factor 2: Overdue weight (higher if number is "due")
            due_score = self.gap.get_due_score(num, 'white')
            overdue_weight = min(due_score, 3.0)  # Cap at 3x

            # Factor 3: Recency bonus (was it in last 10 draws?)
            recent_count = sum(1 for d in self.freq.draws[-10:]
                              if num in d.white_balls)
            recency_weight = 1.0 + (recent_count * 0.1)

            # Combined importance weight (configurable blend)
            # Hot strategy: emphasize frequency
            # Due strategy: emphasize overdue
            # Balanced: equal weight to both
            weights[num] = {
                'frequency': freq_weight,
                'overdue': overdue_weight,
                'recency': recency_weight,
                'balanced': (freq_weight + overdue_weight) / 2,
                'hot_biased': freq_weight * 0.7 + overdue_weight * 0.3,
                'due_biased': freq_weight * 0.3 + overdue_weight * 0.7,
            }

        return weights

    def _compute_pb_weights(self) -> Dict[int, float]:
        """Compute importance weights for Powerballs."""
        weights = {}

        for num in range(1, POWERBALL_MAX + 1):
            freq_count = self.freq.pb_freq.get(num, 1)
            expected = self.freq.expected_frequency('powerball')
            freq_weight = freq_count / expected

            due_score = self.gap.get_due_score(num, 'powerball')
            overdue_weight = min(due_score, 3.0)

            weights[num] = {
                'frequency': freq_weight,
                'overdue': overdue_weight,
                'balanced': (freq_weight + overdue_weight) / 2,
            }

        return weights

    def sample_white_balls(self, strategy: str = 'balanced') -> Tuple[int, ...]:
        """
        Sample 5 white balls using importance sampling.

        Args:
            strategy: 'frequency', 'overdue', 'balanced', 'hot_biased', 'due_biased'
        """
        available = list(range(1, WHITE_BALL_MAX + 1))
        selected = []

        for _ in range(5):
            # Get weights for available numbers
            probs = []
            for num in available:
                w = self.white_weights[num].get(strategy, 1.0)
                probs.append(max(w, 0.01))  # Minimum probability

            # Normalize
            total = sum(probs)
            probs = [p / total for p in probs]

            # Sample
            choice = random.choices(available, weights=probs, k=1)[0]
            selected.append(choice)
            available.remove(choice)

        return tuple(sorted(selected))

    def sample_powerball(self, strategy: str = 'balanced') -> int:
        """Sample Powerball using importance sampling."""
        probs = []
        for num in range(1, POWERBALL_MAX + 1):
            w = self.pb_weights[num].get(strategy, 1.0)
            probs.append(max(w, 0.01))

        total = sum(probs)
        probs = [p / total for p in probs]

        return random.choices(range(1, POWERBALL_MAX + 1), weights=probs, k=1)[0]

    def get_weight_analysis(self, num: int, ball_type: str = 'white') -> Dict:
        """Get detailed weight analysis for a number."""
        if ball_type == 'white':
            return self.white_weights.get(num, {})
        return self.pb_weights.get(num, {})


class NumberGenerator:
    """Generates Powerball numbers using various mathematical strategies."""

    def __init__(self, draws: List[PowerballDraw]):
        self.draws = draws
        self.freq_analyzer = FrequencyAnalyzer(draws)
        self.gap_analyzer = GapAnalyzer(draws)
        self.pattern_analyzer = PatternAnalyzer(draws)

        # Initialize importance sampler
        self.importance_sampler = ImportanceSampler(
            self.freq_analyzer, self.gap_analyzer, self.pattern_analyzer
        )

    def generate_random(self) -> PowerballTicket:
        """Pure random generation."""
        white = tuple(sorted(random.sample(range(1, WHITE_BALL_MAX + 1), 5)))
        pb = random.randint(POWERBALL_MIN, POWERBALL_MAX)
        return PowerballTicket(white, pb, "random")

    def generate_frequency_weighted(self) -> PowerballTicket:
        """Weight selection by historical frequency (hot numbers)."""
        # Build weighted pool for white balls
        weights = []
        for num in range(1, WHITE_BALL_MAX + 1):
            weights.append(self.freq_analyzer.white_freq.get(num, 1))

        # Normalize weights
        total = sum(weights)
        probs = [w / total for w in weights]

        # Select white balls
        white = []
        available = list(range(1, WHITE_BALL_MAX + 1))
        avail_probs = probs.copy()

        for _ in range(5):
            # Renormalize
            p_sum = sum(avail_probs)
            norm_probs = [p / p_sum for p in avail_probs]

            choice = random.choices(available, weights=norm_probs, k=1)[0]
            white.append(choice)

            idx = available.index(choice)
            available.pop(idx)
            avail_probs.pop(idx)

        # Powerball weighted
        pb_weights = [self.freq_analyzer.pb_freq.get(i, 1) for i in range(1, POWERBALL_MAX + 1)]
        pb = random.choices(range(1, POWERBALL_MAX + 1), weights=pb_weights, k=1)[0]

        return PowerballTicket(tuple(sorted(white)), pb, "frequency_weighted")

    def generate_overdue_weighted(self) -> PowerballTicket:
        """Weight selection by how overdue numbers are (gap analysis)."""
        # Get due scores for all numbers
        white_scores = []
        for num in range(1, WHITE_BALL_MAX + 1):
            score = self.gap_analyzer.get_due_score(num, 'white')
            white_scores.append(max(score, 0.1))  # Minimum weight

        # Select white balls
        white = []
        available = list(range(1, WHITE_BALL_MAX + 1))
        avail_scores = white_scores.copy()

        for _ in range(5):
            total = sum(avail_scores)
            probs = [s / total for s in avail_scores]

            choice = random.choices(available, weights=probs, k=1)[0]
            white.append(choice)

            idx = available.index(choice)
            available.pop(idx)
            avail_scores.pop(idx)

        # Powerball
        pb_scores = [max(self.gap_analyzer.get_due_score(i, 'powerball'), 0.1)
                     for i in range(1, POWERBALL_MAX + 1)]
        pb = random.choices(range(1, POWERBALL_MAX + 1), weights=pb_scores, k=1)[0]

        return PowerballTicket(tuple(sorted(white)), pb, "overdue_weighted")

    def generate_hybrid(self) -> PowerballTicket:
        """
        Hybrid strategy combining:
        - 2 hot numbers (frequent)
        - 2 overdue numbers (due)
        - 1 random number
        """
        hot = [n for n, _ in self.freq_analyzer.get_hot_numbers(15)]
        overdue = [n for n, _, _ in self.gap_analyzer.get_overdue_numbers(15)]

        white = set()

        # 2 hot numbers
        hot_available = [n for n in hot if n not in white]
        white.update(random.sample(hot_available[:10], min(2, len(hot_available))))

        # 2 overdue numbers
        overdue_available = [n for n in overdue if n not in white]
        white.update(random.sample(overdue_available[:10], min(2, len(overdue_available))))

        # Fill remaining with random
        while len(white) < 5:
            n = random.randint(1, WHITE_BALL_MAX)
            if n not in white:
                white.add(n)

        # Powerball - mix of hot and overdue (filter to valid range 1-26)
        if random.random() < 0.5:
            hot_pb = [n for n, _ in self.freq_analyzer.get_hot_numbers(10, 'powerball') if n <= POWERBALL_MAX]
            pb = random.choice(hot_pb) if hot_pb else random.randint(1, POWERBALL_MAX)
        else:
            overdue_pb = [n for n, _, _ in self.gap_analyzer.get_overdue_numbers(10, 'powerball') if n <= POWERBALL_MAX]
            pb = random.choice(overdue_pb) if overdue_pb else random.randint(1, POWERBALL_MAX)

        return PowerballTicket(tuple(sorted(white)), pb, "hybrid")

    def generate_pattern_optimized(self) -> PowerballTicket:
        """Generate numbers matching optimal historical patterns."""
        patterns = self.pattern_analyzer.get_optimal_patterns()

        # Target even/odd ratio
        target_evens = patterns['even_count'][0][0]
        target_sum = patterns['avg_sum']
        sum_tolerance = 30

        # Generate until we match patterns
        for _ in range(1000):
            white = random.sample(range(1, WHITE_BALL_MAX + 1), 5)
            evens = sum(1 for n in white if n % 2 == 0)
            total = sum(white)

            if evens == target_evens and abs(total - target_sum) <= sum_tolerance:
                pb = random.randint(1, POWERBALL_MAX)
                return PowerballTicket(tuple(sorted(white)), pb, "pattern_optimized")

        # Fallback to random if can't match patterns
        return self.generate_random()

    def generate_pair_based(self) -> PowerballTicket:
        """Generate using historically common pairs."""
        hot_pairs = self.freq_analyzer.get_hot_pairs(20)

        white = set()

        # Start with a hot pair
        if hot_pairs:
            pair = random.choice(hot_pairs[:5])[0]
            white.update(pair)

        # Add more numbers, preferring those that pair well with existing
        while len(white) < 5:
            candidates = []
            for n in range(1, WHITE_BALL_MAX + 1):
                if n not in white:
                    # Score based on pairing history
                    score = sum(self.freq_analyzer.pair_freq.get((min(n, w), max(n, w)), 0)
                               for w in white)
                    candidates.append((n, score + 1))  # +1 to avoid zero weights

            if candidates:
                nums, weights = zip(*candidates)
                choice = random.choices(nums, weights=weights, k=1)[0]
                white.add(choice)
            else:
                break

        # Filter to valid powerball range (1-26)
        hot_pb = [n for n, _ in self.freq_analyzer.get_hot_numbers(15, 'powerball') if n <= POWERBALL_MAX]
        pb = random.choice(hot_pb) if hot_pb else random.randint(1, POWERBALL_MAX)

        return PowerballTicket(tuple(sorted(white)), pb, "pair_based")

    def generate_importance_sampled(self, strategy: str = 'balanced') -> PowerballTicket:
        """
        Generate numbers using proper Importance Sampling.

        This is the mathematically rigorous approach:
        - P(n) = w(n) / Σw(i) where w(n) is the importance weight

        Available strategies:
        - 'balanced': Equal weight to frequency and overdue factors
        - 'frequency': Bias toward historically hot numbers
        - 'overdue': Bias toward numbers that are "due"
        - 'hot_biased': 70% frequency, 30% overdue
        - 'due_biased': 30% frequency, 70% overdue

        Mathematical insight: By weighting toward observed distributions,
        we reduce variance in the number of draws needed to match a
        historical pattern.
        """
        white = self.importance_sampler.sample_white_balls(strategy)
        pb = self.importance_sampler.sample_powerball(strategy)
        return PowerballTicket(white, pb, f"importance_{strategy}")

    def generate_tickets(self, count: int, strategy: str = 'mixed') -> List[PowerballTicket]:
        """Generate multiple tickets using specified strategy."""
        strategies = {
            'random': self.generate_random,
            'frequency': self.generate_frequency_weighted,
            'overdue': self.generate_overdue_weighted,
            'hybrid': self.generate_hybrid,
            'pattern': self.generate_pattern_optimized,
            'pairs': self.generate_pair_based,
            'importance': lambda: self.generate_importance_sampled('balanced'),
            'importance_hot': lambda: self.generate_importance_sampled('hot_biased'),
            'importance_due': lambda: self.generate_importance_sampled('due_biased'),
        }

        tickets = []

        if strategy == 'mixed':
            # Mix of all strategies
            strat_list = list(strategies.values())
            for i in range(count):
                gen = strat_list[i % len(strat_list)]
                tickets.append(gen())
        elif strategy in strategies:
            gen = strategies[strategy]
            for _ in range(count):
                tickets.append(gen())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return tickets


class MatchSimulator:
    """Simulates matching tickets against historical/generated draws."""

    PRIZE_TIERS = {
        (5, True): "JACKPOT",
        (5, False): "$1,000,000",
        (4, True): "$50,000",
        (4, False): "$100",
        (3, True): "$100",
        (3, False): "$7",
        (2, True): "$7",
        (1, True): "$4",
        (0, True): "$4",
    }

    def __init__(self, historical_draws: List[PowerballDraw]):
        self.historical = historical_draws

    def find_match_in_history(self, ticket: PowerballTicket) -> Optional[MatchResult]:
        """Check if ticket matches any historical draw."""
        for idx, draw in enumerate(self.historical):
            white_match, pb_match = draw.matches(ticket)
            if white_match == 5 and pb_match:
                return MatchResult(
                    ticket=ticket,
                    attempts=idx + 1,
                    matched_draw=draw,
                    prize_tier="JACKPOT (Historical)"
                )
        return None

    def simulate_until_jackpot(self, ticket: PowerballTicket, max_attempts: int = 100_000_000) -> MatchResult:
        """Simulate random draws until jackpot is hit."""
        partial_matches = defaultdict(int)

        for attempt in range(1, max_attempts + 1):
            # Generate random draw
            white = tuple(sorted(random.sample(range(1, WHITE_BALL_MAX + 1), 5)))
            pb = random.randint(1, POWERBALL_MAX)

            # Check match
            white_match = len(set(white) & set(ticket.white_balls))
            pb_match = pb == ticket.powerball

            # Track partial matches
            key = f"{white_match}+{'PB' if pb_match else 'X'}"
            partial_matches[key] += 1

            if white_match == 5 and pb_match:
                draw = PowerballDraw(datetime.now(), white, pb)
                return MatchResult(
                    ticket=ticket,
                    attempts=attempt,
                    matched_draw=draw,
                    prize_tier="JACKPOT",
                    partial_matches=dict(partial_matches)
                )

            # Progress indicator
            if attempt % 10_000_000 == 0:
                print(f"  Attempt {attempt:,}... still searching")

        return MatchResult(
            ticket=ticket,
            attempts=max_attempts,
            partial_matches=dict(partial_matches)
        )

    def evaluate_tickets(self, tickets: List[PowerballTicket],
                        target_draw: PowerballDraw) -> List[Tuple[PowerballTicket, int, bool, str]]:
        """Evaluate tickets against a specific draw."""
        results = []
        for ticket in tickets:
            white_match, pb_match = target_draw.matches(ticket)
            prize = self.PRIZE_TIERS.get((white_match, pb_match), "No prize")
            results.append((ticket, white_match, pb_match, prize))
        return results


def load_historical_data(filepath: str) -> List[PowerballDraw]:
    """Load historical draws from CSV file."""
    draws = []

    with open(filepath, 'r') as f:
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
            except (ValueError, KeyError) as e:
                continue

    # Sort by date
    draws.sort(key=lambda x: x.date)
    return draws


def print_analysis_report(draws: List[PowerballDraw]):
    """Print comprehensive analysis of historical data."""
    freq = FrequencyAnalyzer(draws)
    gap = GapAnalyzer(draws)
    pattern = PatternAnalyzer(draws)

    print("\n" + "="*70)
    print("POWERBALL MATHEMATICAL ANALYSIS REPORT")
    print("="*70)
    print(f"Total draws analyzed: {len(draws):,}")
    print(f"Date range: {draws[0].date.strftime('%m/%d/%Y')} - {draws[-1].date.strftime('%m/%d/%Y')}")
    print(f"Jackpot odds: 1 in {JACKPOT_ODDS:,}")

    print("\n" + "-"*70)
    print("FREQUENCY ANALYSIS")
    print("-"*70)

    print("\n[HOT WHITE BALLS] (Most Frequent):")
    for num, count in freq.get_hot_numbers(10):
        expected = freq.expected_frequency('white')
        diff = ((count - expected) / expected) * 100
        print(f"  #{num:2d}: {count:4d} times ({diff:+.1f}% vs expected)")

    print("\n[COLD WHITE BALLS] (Least Frequent):")
    for num, count in freq.get_cold_numbers(10):
        expected = freq.expected_frequency('white')
        diff = ((count - expected) / expected) * 100
        print(f"  #{num:2d}: {count:4d} times ({diff:+.1f}% vs expected)")

    print("\n[HOT POWERBALLS]:")
    for num, count in freq.get_hot_numbers(5, 'powerball'):
        expected = freq.expected_frequency('powerball')
        diff = ((count - expected) / expected) * 100
        print(f"  #{num:2d}: {count:4d} times ({diff:+.1f}% vs expected)")

    print("\n[COLD POWERBALLS]:")
    for num, count in freq.get_cold_numbers(5, 'powerball'):
        expected = freq.expected_frequency('powerball')
        diff = ((count - expected) / expected) * 100
        print(f"  #{num:2d}: {count:4d} times ({diff:+.1f}% vs expected)")

    print("\n" + "-"*70)
    print("GAP ANALYSIS (Overdue Numbers)")
    print("-"*70)

    print("\n[OVERDUE WHITE BALLS]:")
    for num, gap, ratio in gap.get_overdue_numbers(10):
        status = "VERY OVERDUE" if ratio > 2 else "Overdue" if ratio > 1.5 else "Slightly overdue"
        print(f"  #{num:2d}: {gap:3d} draws since last ({ratio:.2f}x avg gap) - {status}")

    print("\n[OVERDUE POWERBALLS]:")
    for num, gap, ratio in gap.get_overdue_numbers(5, 'powerball'):
        status = "VERY OVERDUE" if ratio > 2 else "Overdue" if ratio > 1.5 else "Slightly overdue"
        print(f"  #{num:2d}: {gap:3d} draws since last ({ratio:.2f}x avg gap) - {status}")

    print("\n" + "-"*70)
    print("PATTERN ANALYSIS")
    print("-"*70)

    patterns = pattern.get_optimal_patterns()
    print(f"\n[EVEN/ODD Distribution] (Most common patterns):")
    for count, freq_count in patterns['even_count']:
        print(f"  {count} even, {5-count} odd: {freq_count} times ({freq_count/len(draws)*100:.1f}%)")

    print(f"\n[HIGH/LOW Distribution] (1-34 low, 35-69 high):")
    for count, freq_count in patterns['high_count']:
        print(f"  {count} high, {5-count} low: {freq_count} times ({freq_count/len(draws)*100:.1f}%)")

    print(f"\n[SUM Statistics]:")
    print(f"  Average sum: {patterns['avg_sum']:.1f}")
    print(f"  Sum range: {patterns['sum_range'][0]} - {patterns['sum_range'][1]}")
    print(f"  Optimal target: {int(patterns['avg_sum'] - 30)} - {int(patterns['avg_sum'] + 30)}")

    print("\n[HOT PAIRS] (Numbers that appear together):")
    for pair, count in freq.get_hot_pairs(5):
        print(f"  {pair[0]:2d} & {pair[1]:2d}: {count:3d} times together")

    print("\n" + "="*70)


def main():
    """Main interactive program."""
    import os

    # Find data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'historical_data.csv')

    if not os.path.exists(data_file):
        print("ERROR: historical_data.csv not found!")
        print("Please ensure the data file is in the same directory.")
        return

    print("\n" + "="*70)
    print("   POWERBALL MATHEMATICAL ANALYZER & GENERATOR")
    print("   'Finding patterns in randomness'")
    print("="*70)

    # Load data
    print("\nLoading historical data...")
    draws = load_historical_data(data_file)
    print(f"Loaded {len(draws):,} historical draws")

    # Initialize components
    generator = NumberGenerator(draws)
    simulator = MatchSimulator(draws)

    while True:
        print("\n" + "-"*70)
        print("MENU")
        print("-"*70)
        print("1. View Analysis Report")
        print("2. Generate Tickets")
        print("3. Run Jackpot Simulation")
        print("4. Test Against Recent Draw")
        print("5. Quick Pick Comparison")
        print("6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == '1':
            print_analysis_report(draws)

        elif choice == '2':
            try:
                count = int(input("How many tickets? ($2 each): "))
                if count <= 0:
                    print("Invalid number")
                    continue

                cost = count * 2
                print(f"\nTotal cost: ${cost}")

                print("\nStrategies available:")
                print("  1. mixed    - Combination of all strategies")
                print("  2. random   - Pure random selection")
                print("  3. frequency - Weighted toward hot numbers")
                print("  4. overdue  - Weighted toward overdue numbers")
                print("  5. hybrid   - Mix of hot + overdue + random")
                print("  6. pattern  - Match historical patterns")
                print("  7. pairs    - Based on common pairs")

                strat = input("Strategy (1-7 or name): ").strip()
                strat_map = {'1': 'mixed', '2': 'random', '3': 'frequency',
                            '4': 'overdue', '5': 'hybrid', '6': 'pattern', '7': 'pairs'}
                strategy = strat_map.get(strat, strat)

                tickets = generator.generate_tickets(count, strategy)

                print(f"\nGenerated {len(tickets)} tickets:\n")
                for i, ticket in enumerate(tickets, 1):
                    print(f"  Ticket #{i}: {ticket}")

            except ValueError as e:
                print(f"Error: {e}")

        elif choice == '3':
            print("\n[JACKPOT SIMULATION]")
            print("This simulates how many random draws it takes to hit the jackpot.")
            print(f"Mathematical odds: 1 in {JACKPOT_ODDS:,}")
            print("\nWARNING: This can take a VERY long time!")

            strat = input("Strategy for ticket (random/frequency/hybrid): ").strip() or 'hybrid'
            ticket = generator.generate_tickets(1, strat)[0]
            print(f"\nYour ticket: {ticket}")

            max_att = input("Max attempts (default 10M): ").strip()
            max_attempts = int(max_att) if max_att else 10_000_000

            print(f"\nSimulating up to {max_attempts:,} draws...")
            result = simulator.simulate_until_jackpot(ticket, max_attempts)

            if result.matched_draw:
                print(f"\n{'='*50}")
                print(f"JACKPOT HIT after {result.attempts:,} attempts!")
                print(f"{'='*50}")
                print(f"Ticket: {result.ticket}")
                print(f"Draw:   {result.matched_draw}")

                if result.attempts > 0:
                    cost = result.attempts * 2
                    print(f"\nCost to win: ${cost:,}")
                    print(f"That's {result.attempts / JACKPOT_ODDS * 100:.6f}x the expected attempts")
            else:
                print(f"\nNo jackpot after {max_attempts:,} attempts")

            if result.partial_matches:
                print("\nPartial matches during simulation:")
                for match, count in sorted(result.partial_matches.items()):
                    print(f"  {match}: {count:,} times")

        elif choice == '4':
            print("\n[TEST AGAINST RECENT DRAW]")
            recent = draws[-1]
            print(f"Most recent draw: {recent}")

            count = int(input("How many tickets to test? "))
            strat = input("Strategy (mixed/random/frequency/hybrid): ").strip() or 'mixed'

            tickets = generator.generate_tickets(count, strat)
            results = simulator.evaluate_tickets(tickets, recent)

            print(f"\nResults against {recent.date.strftime('%m/%d/%Y')}:")
            winners = []
            for ticket, white, pb, prize in results:
                if prize != "No prize":
                    winners.append((ticket, white, pb, prize))

            if winners:
                print(f"\nWINNERS ({len(winners)}):")
                for ticket, white, pb, prize in winners:
                    pb_str = "+" if pb else ""
                    print(f"  {ticket} -> {white}{pb_str} white = {prize}")
            else:
                print("No winners this time")

            print(f"\nTotal cost: ${count * 2}")

        elif choice == '5':
            print("\n[QUICK PICK COMPARISON]")
            print("Compare 1000 tickets from each strategy against recent draw")

            recent = draws[-1]
            print(f"Testing against: {recent}\n")

            strategies = ['random', 'frequency', 'overdue', 'hybrid', 'pattern', 'pairs']

            for strat in strategies:
                tickets = generator.generate_tickets(1000, strat)
                results = simulator.evaluate_tickets(tickets, recent)

                wins = sum(1 for _, _, _, prize in results if prize != "No prize")
                print(f"  {strat:15s}: {wins} wins out of 1000")

        elif choice == '6':
            print("\nGood luck!")
            break

        else:
            print("Invalid option")


if __name__ == "__main__":
    main()
