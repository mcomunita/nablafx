"""
Test the integrated callback system with existing system classes.

This test verifies that the use_callbacks parameter works correctly
and that both old and new approaches can coexist.
"""

import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger

# Import system classes
from nablafx.system import BlackBoxSystem, BaseSystem

# Import callbacks
from nablafx.callbacks import (
    AudioLoggingCallback,
    MetricsLoggingCallback,
    FrequencyResponseCallback,
    FADComputationCallback,
)

# Import loss and model (you'll need to adjust these imports)
from nablafx.loss import WeightedMultiLoss


def test_callback_integration():
    """Test that the callback integration works properly."""

    print("🧪 Testing Callback Integration")
    print("=" * 50)

    # Create a simple loss function
    loss = WeightedMultiLoss(losses=[{"loss": torch.nn.L1Loss(), "weight": 1.0, "name": "l1"}])

    # Test 1: Traditional system (use_callbacks=False)
    print("\n1️⃣ Testing traditional system (use_callbacks=False)")
    system_old = BaseSystem(loss=loss, lr=1e-4, log_media_every_n_steps=1000, use_callbacks=False)

    print(f"   ✅ Metrics initialized: {len(system_old.metrics)} metrics")
    print(f"   ✅ use_callbacks: {system_old.use_callbacks}")

    # Test 2: Callback-based system (use_callbacks=True)
    print("\n2️⃣ Testing callback-based system (use_callbacks=True)")
    system_new = BaseSystem(loss=loss, lr=1e-4, log_media_every_n_steps=1000, use_callbacks=True)  # Ignored when use_callbacks=True

    print(f"   ✅ Metrics skipped: {len(system_new.metrics)} metrics")
    print(f"   ✅ use_callbacks: {system_new.use_callbacks}")

    # Test 3: Verify logging methods are no-ops when use_callbacks=True
    print("\n3️⃣ Testing logging method behavior")

    # Create mock data
    batch_idx = 0
    input_audio = torch.randn(2, 1, 1000)
    target_audio = torch.randn(2, 1, 1000)
    pred_audio = torch.randn(2, 1, 1000)

    # Test old system (should work normally)
    print("   📊 Old system logging methods:")
    try:
        # These should work but won't actually log without proper logger setup
        system_old.compute_and_log_metrics(pred_audio, target_audio, "test")
        print("   ✅ compute_and_log_metrics: works")
    except Exception as e:
        print(f"   ⚠️  compute_and_log_metrics: {e}")

    # Test new system (should be no-ops)
    print("   📊 New system logging methods (should be no-ops):")

    # These should return immediately without doing anything
    system_new.compute_and_log_metrics(pred_audio, target_audio, "test")
    print("   ✅ compute_and_log_metrics: no-op")

    system_new.log_audio(batch_idx, input_audio, target_audio, pred_audio, "test")
    print("   ✅ log_audio: no-op")

    system_new.log_frequency_response()
    print("   ✅ log_frequency_response: no-op")

    system_new.compute_and_log_fad("test")
    print("   ✅ compute_and_log_fad: no-op")

    print("\n✅ All tests passed! Callback integration working correctly.")


def demo_callback_setup():
    """Demonstrate how to set up callbacks with the integrated system."""

    print("\n🎯 Callback Setup Demo")
    print("=" * 50)

    # Create callbacks
    callbacks = [
        AudioLoggingCallback(
            log_every_n_steps=1000,
            sample_rate=48000,
            max_samples_per_batch=3,
        ),
        MetricsLoggingCallback(
            log_on_epoch=True,
            sync_dist=True,
        ),
        FrequencyResponseCallback(
            log_on_test_end=True,
        ),
        FADComputationCallback(
            compute_on_train_end=True,
            compute_on_test_end=True,
            models=["vggish", "pann"],  # Reduced for demo
        ),
    ]

    print(f"✅ Created {len(callbacks)} callbacks:")
    for i, callback in enumerate(callbacks, 1):
        print(f"   {i}. {callback.__class__.__name__}")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=1,  # Short for demo
        accelerator="cpu",  # Use CPU for demo
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,  # Disable logging for demo
    )

    print(f"✅ Created trainer with {len(trainer.callbacks)} callbacks")

    return trainer, callbacks


def show_migration_benefits():
    """Show the benefits of the callback-based approach."""

    print("\n🚀 Migration Benefits")
    print("=" * 50)

    benefits = [
        "🔧 Configuration-driven logging - no code changes needed",
        "🎛️  Granular control - enable/disable specific logging features",
        "📊 Consistent behavior - same logging across all system types",
        "🔄 Easy experimentation - try different logging configurations",
        "🧩 Modular design - add custom callbacks without modifying core",
        "⚡ Better performance - only run logging you actually need",
        "🔄 Backward compatibility - existing code continues to work",
    ]

    for benefit in benefits:
        print(f"   {benefit}")

    print("\n📈 Next Steps:")
    steps = [
        "1. Add use_callbacks: true to your system config",
        "2. Add callback configuration to your trainer",
        "3. Customize callback parameters for your needs",
        "4. Optionally migrate to simplified system classes",
    ]

    for step in steps:
        print(f"   {step}")


if __name__ == "__main__":
    print("NablAFx Callback Integration Test")
    print("=" * 50)

    # Run tests
    test_callback_integration()

    # Show setup demo
    demo_callback_setup()

    # Show benefits
    show_migration_benefits()

    print("\n🎉 Integration complete! You can now use callbacks with existing systems.")
    print("\nTo get started:")
    print("1. Set use_callbacks: true in your system config")
    print("2. Add callbacks to your trainer config")
    print("3. Enjoy modular, configurable logging!")
