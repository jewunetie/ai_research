"""Integration tests for the simulation"""

import pytest
from src.config.config_loader import load_config, validate_config


def test_config_loading():
    """Test that default config loads and validates correctly"""
    config = load_config()

    assert config is not None
    assert 'simulation' in config
    assert 'agents' in config
    assert 'marketplace' in config
    assert 'revenue' in config

    # Validate config
    assert validate_config(config) is True


def test_config_structure():
    """Test that config has required fields"""
    config = load_config()

    # Simulation config
    assert 'random_seed' in config['simulation']
    assert 'num_steps' in config['simulation']
    assert config['simulation']['random_seed'] == 42
    assert config['simulation']['num_steps'] == 10000

    # Agent config
    assert 'users' in config['agents']
    assert 'creators' in config['agents']
    assert 'companies' in config['agents']

    # User config
    assert 'count' in config['agents']['users']
    assert 'distribution' in config['agents']['users']

    # Revenue config
    assert 'platform_share' in config['revenue']
    assert 'creator_share' in config['revenue']
    assert 'user_share' in config['revenue']


def test_revenue_shares_sum_to_one():
    """Test that revenue shares sum to approximately 1.0"""
    config = load_config()

    total_share = (
        config['revenue']['platform_share'] +
        config['revenue']['creator_share'] +
        config['revenue']['user_share']
    )

    assert abs(total_share - 1.0) < 0.01  # Allow small floating point error


def test_user_distribution_sums_to_one():
    """Test that user type distribution sums to approximately 1.0"""
    config = load_config()

    user_dist = config['agents']['users']['distribution']
    total = sum(user_dist.values())

    assert abs(total - 1.0) < 0.01  # Allow small floating point error


def test_creator_distribution_sums_to_one():
    """Test that creator size distribution sums to approximately 1.0"""
    config = load_config()

    creator_dist = config['agents']['creators']['distribution']
    total = sum(creator_dist.values())

    assert abs(total - 1.0) < 0.01  # Allow small floating point error
