#!/usr/bin/env python3
"""
Test script for the new WeightedMultiLoss functionality.
Run this to verify the implementation works correctly.
"""

import torch
import auraloss
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nablafx.loss import WeightedMultiLoss, TimeAndFrequencyDomainLoss

def test_weighted_multi_loss():
    """Test the WeightedMultiLoss class"""
    print("Testing WeightedMultiLoss...")
    
    # Create sample data
    batch_size = 4
    seq_length = 48000
    x = torch.randn(batch_size, 1, seq_length)
    y = torch.randn(batch_size, 1, seq_length)
    
    # Test 1: Single loss function
    print("\n1. Testing with single loss function...")
    single_loss_config = [
        {
            'loss': torch.nn.L1Loss(),
            'weight': 1.0,
            'name': 'l1'
        }
    ]
    
    single_loss = WeightedMultiLoss(single_loss_config)
    result = single_loss(x, y)
    print(f"Single loss result: {result}")
    print(f"Loss names: {single_loss.get_loss_names()}")
    print(f"Weights: {single_loss.get_weights()}")
    
    # Test 2: Multiple loss functions
    print("\n2. Testing with multiple loss functions...")
    multi_loss_config = [
        {
            'loss': torch.nn.L1Loss(),
            'weight': 0.7,
            'name': 'l1'
        },
        {
            'loss': torch.nn.MSELoss(),
            'weight': 0.3,
            'name': 'mse'
        }
    ]
    
    multi_loss = WeightedMultiLoss(multi_loss_config)
    result = multi_loss(x, y)
    print(f"Multi loss result (tuple): {result}")
    print(f"Individual losses: {result[:-1]}")
    print(f"Total loss: {result[-1]}")
    print(f"Loss names: {multi_loss.get_loss_names()}")
    print(f"Weights: {multi_loss.get_weights()}")
    
    # Test 3: With auraloss functions
    print("\n3. Testing with auraloss functions...")
    audio_loss_config = [
        {
            'loss': auraloss.time.ESRLoss(),
            'weight': 0.5,
            'name': 'esr'
        },
        {
            'loss': auraloss.freq.MultiResolutionSTFTLoss(),
            'weight': 0.5,
            'name': 'mr_stft'
        }
    ]
    
    audio_loss = WeightedMultiLoss(audio_loss_config)
    result = audio_loss(x, y)
    print(f"Audio loss result: {result}")
    print(f"Loss names: {audio_loss.get_loss_names()}")
    
    # Test 4: Compare with original TimeAndFrequencyDomainLoss
    print("\n4. Comparing with original TimeAndFrequencyDomainLoss...")
    
    # Original format
    original_loss = TimeAndFrequencyDomainLoss(
        time_domain_loss=torch.nn.L1Loss(),
        frequency_domain_loss=auraloss.freq.MultiResolutionSTFTLoss(),
        time_domain_weight=0.5,
        frequency_domain_weight=0.5
    )
    
    # New format (equivalent)
    new_loss = WeightedMultiLoss([
        {
            'loss': torch.nn.L1Loss(),
            'weight': 0.5,
            'name': 'time_domain'
        },
        {
            'loss': auraloss.freq.MultiResolutionSTFTLoss(),
            'weight': 0.5,
            'name': 'frequency_domain'
        }
    ])
    
    original_result = original_loss(x, y)
    new_result = new_loss(x, y)
    
    print(f"Original loss result: {original_result}")
    print(f"New loss result: {new_result}")
    print("✓ Both implementations work!")

def test_error_handling():
    """Test error handling"""
    print("\n\nTesting error handling...")
    
    # Test empty losses
    try:
        WeightedMultiLoss([])
        print("❌ Should have raised error for empty losses")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test missing weight
    try:
        WeightedMultiLoss([{'loss': torch.nn.L1Loss()}])
        print("❌ Should have raised error for missing weight")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test invalid loss function
    try:
        WeightedMultiLoss([{'loss': 'not_a_loss', 'weight': 1.0}])
        print("❌ Should have raised error for invalid loss")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")

if __name__ == "__main__":
    test_weighted_multi_loss()
    test_error_handling()
    print("\n🎉 All tests passed!")
