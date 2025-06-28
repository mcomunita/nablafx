# NablAFx Updates ### Developer experience

---

## Immediate Priority: Loss Functions & Logging 🔥

### Why These Are Critical
The current hardcoded loss instantiation and embedded logging in system classes limit:

- **Experimental flexibility** - Researchers can't easily try different loss combinations
- **Code reusability** - Logging logic is tightly coupled to specific system classes
- **Configuration transparency** - Loss configurations are scattered across code and configs
- **Callback extensibility** - Adding new logging features requires modifying core classes

### Quick Implementation Plan (Weeks 1-2)

#### Week 1: Loss Function System
1. **Create `nablafx.loss.registry`** - Registry for discovering loss functions
2. **Implement `WeightedMultiLoss`** - Base class for combining multiple loss functions
3. **Add loss validation** - Check compatibility and parameter constraints
4. **Update existing configurations** - Migrate current loss setups to new system

#### Week 2: Callback System  
1. **Extract logging methods** from system.py into callback classes
2. **Create base callback interface** - Common functionality and lifecycle hooks
3. **Update trainer configurations** - Enable callback specification in YAML
4. **Add callback examples** - Documentation and configuration templates

### Expected Impact
- **Faster experimentation** - New loss functions configurable without code changes
- **Cleaner architecture** - Separation of concerns between training logic and logging
- **Better reproducibility** - Complete experimental setup captured in configuration
- **Easier contribution** - Community can add callbacks without touching core training loop

---dmap

## Overview
This document outlines a comprehensive roadmap for improving the modularity, maintainability, and overall architecture of the NablAFx framework for differentiable audio effects modeling.

## Current State Analysis

### Strengths
- Well-structured black-box and gray-box modeling framework
- Clean separation between processors, controllers, and models
- Good integration with PyTorch Lightning and Weights & Biases
- Comprehensive examples and configurations

### Areas for Improvement
- Code organization and modularity
- Documentation and type safety
- Testing infrastructure
- Package structure and distribution
- Developer experience

---

## 1. Package Structure & Organization 🏗️

### Priority: HIGH
### Timeline: 2-3 weeks

#### 1.1 Restructure Core Modules
- **Separate concerns into subpackages**:
  ```
  nablafx/
  ├── __init__.py
  ├── core/
  │   ├── __init__.py
  │   ├── models.py
  │   ├── interfaces.py
  │   └── system.py
  ├── processors/
  │   ├── __init__.py
  │   ├── base.py
  │   ├── dsp/
  │   ├── neural/
  │   └── hybrid/
  ├── controllers/
  │   ├── __init__.py
  │   ├── base.py
  │   ├── static.py
  │   └── dynamic.py
  ├── data/
  │   ├── __init__.py
  │   ├── datasets.py
  │   └── transforms.py
  ├── models/
  │   ├── __init__.py
  │   ├── architectures/
  │   └── losses/
  └── utils/
      ├── __init__.py
      ├── plotting.py
      ├── dsp.py
      └── metrics.py
  ```

#### 1.2 Create Clear API Boundaries
- Define public APIs in `__init__.py` files
- Implement factory patterns for model creation
- Add configuration validation schemas

---

## 2. Type Safety & Documentation 📝

### Priority: HIGH
### Timeline: 2-3 weeks

#### 2.1 Add Comprehensive Type Hints
- Add type hints to all public methods and functions
- Use generics for better type safety in model definitions
- Implement protocols for processor and controller interfaces

#### 2.2 Documentation Improvements
- Add comprehensive docstrings following NumPy/Google style
- Create API reference documentation with Sphinx
- Add inline code examples in docstrings
- Create architectural decision records (ADRs)

#### 2.3 Configuration Schema
- Implement Pydantic models for configuration validation
- Replace manual parameter validation with schema validation
- Add configuration documentation and examples

---

## 3. Testing Infrastructure 🧪

### Priority: HIGH
### Timeline: 1-2 weeks

#### 3.1 Comprehensive Test Suite
- **Unit tests** for all core components:
  - Processors (DSP operations, neural networks)
  - Controllers (parameter mapping)
  - Models (forward pass, state management)
  - Data loading and preprocessing
- **Integration tests** for complete workflows
- **Performance tests** for real-time capability validation

#### 3.2 Test Organization
```
tests/
├── unit/
│   ├── test_processors/
│   ├── test_controllers/
│   ├── test_models/
│   └── test_data/
├── integration/
│   ├── test_training/
│   ├── test_inference/
│   └── test_configs/
├── performance/
└── fixtures/
```

#### 3.3 Continuous Integration
- Set up GitHub Actions for automated testing
- Add code coverage reporting
- Implement pre-commit hooks for code quality

---

## 4. Configuration Management 🔧

### Priority: MEDIUM
### Timeline: 1-2 weeks

#### 4.1 Unified Configuration System
- Replace multiple YAML files with hierarchical configuration
- Implement configuration inheritance and composition
- Add configuration validation and error reporting

#### 4.2 Configuration Templates
- Create template configurations for common use cases
- Add configuration discovery and auto-completion
- Implement configuration migration tools

---

## 5. Processor Architecture Refactoring ⚙️

### Priority: MEDIUM-HIGH
### Timeline: 2-3 weeks

#### 5.1 Base Processor Interface
- Define abstract base classes for all processor types
- Standardize the processor API (forward, reset_states, etc.)
- Implement proper inheritance hierarchy

#### 5.2 DSP Processor Separation
- Move DSP-specific code to dedicated submodule
- Separate differentiable DSP from neural processors
- Add DSP parameter validation and constraints

#### 5.3 Plugin Architecture
- Implement processor discovery and registration
- Allow dynamic loading of custom processors
- Add processor metadata and capability reporting

---

## 6. Model Factory & Builder Patterns 🏭

### Priority: MEDIUM
### Timeline: 1-2 weeks

#### 6.1 Model Factory
- Implement factory pattern for model creation from configs
- Add model validation and compatibility checking
- Support for model composition and chaining

#### 6.2 Builder Pattern for Complex Models
- Create fluent API for model construction
- Support conditional model building
- Add model architecture visualization

---

## 7. Loss Function Configuration System 🎯

### Priority: HIGH
### Timeline: 1-2 weeks

#### 7.1 Configurable Loss Functions
- **Create loss function registry** for easy discovery and instantiation
- **Support complex loss compositions** via YAML configuration
- **Add validation for loss function parameters** and compatibility
- **Enable hot-swapping of loss functions** during experimentation

#### 7.2 Loss Function Factory
- Implement factory pattern for loss function creation from configs
- Support nested loss functions (e.g., weighted combinations)
- Add loss function metadata and documentation
- Create loss function templates for common use cases

Example configuration:
```yaml
loss:
  class_path: nablafx.loss.WeightedMultiLoss
  init_args:
    losses:
      - name: "time_domain"
        weight: 0.7
        loss:
          class_path: auraloss.time.ESRLoss
      - name: "frequency_domain" 
        weight: 0.3
        loss:
          class_path: auraloss.freq.MultiResolutionSTFTLoss
          init_args:
            fft_sizes: [1024, 2048, 512]
```

---

## 8. Callback-Based Logging System 📈

### Priority: HIGH  
### Timeline: 1-2 weeks

#### 8.1 Convert System Logging to Callbacks
- **Extract logging functions from system.py** into dedicated callback classes:
  - `AudioLoggingCallback` - for audio sample logging
  - `MetricsLoggingCallback` - for metrics computation and logging
  - `FrequencyResponseCallback` - for frequency response visualization
  - `FADComputationCallback` - for Fréchet Audio Distance computation
  - `ParameterVisualizationCallback` - for gray-box parameter logging

#### 8.2 Configurable Callback System
- **Enable callback configuration in trainer YAML**:
```yaml
trainer:
  callbacks:
    - class_path: nablafx.callbacks.AudioLoggingCallback
      init_args:
        log_every_n_steps: 1000
        sample_rate: 48000
        max_samples_per_batch: 5
    - class_path: nablafx.callbacks.MetricsLoggingCallback
      init_args:
        compute_fad: true
        fad_every_n_epochs: 5
    - class_path: nablafx.callbacks.FrequencyResponseCallback
      init_args:
        log_every_n_epochs: 10
        frequency_range: [20, 20000]
```

#### 8.3 Modular Logging Architecture
- **Create base callback classes** with common functionality
- **Support conditional logging** based on training phase and metrics
- **Add callback dependencies** and execution order management
- **Enable callback state persistence** across training sessions

---

## 9. Data Pipeline Improvements 📊

### Priority: MEDIUM
### Timeline: 1-2 weeks

#### 9.1 Data Transforms
- Separate data loading from preprocessing
- Implement composable transform pipeline
- Add data validation and quality checks

#### 9.2 Dataset Abstraction
- Create generic dataset interface
- Support multiple audio formats and sources
- Add dataset metadata and statistics

---

## 10. Performance Optimization 🚀

### Priority: MEDIUM-LOW
### Timeline: 2-3 weeks

#### 10.1 Memory Optimization
- Implement memory-efficient data loading
- Add gradient checkpointing for large models
- Optimize tensor operations and reduce copies

#### 10.2 Computation Optimization
- Profile critical paths and optimize bottlenecks
- Add mixed precision training support
- Implement efficient batch processing

---

## 11. Developer Experience 👨‍💻

### Priority: MEDIUM
### Timeline: 1-2 weeks

#### 11.1 CLI Improvements
- Create intuitive command-line interface
- Add interactive configuration setup
- Implement progress bars and status reporting

#### 11.2 Debugging Tools
- Add model inspection utilities
- Implement tensor debugging helpers
- Create visualization tools for model internals

---

## 12. Distribution & Packaging 📦

### Priority: LOW-MEDIUM
### Timeline: 1 week

#### 12.1 Package Distribution
- Set up proper Python packaging with `pyproject.toml`
- Create conda packages for easier installation
- Add dependency management and version pinning

#### 12.2 Installation Scripts
- Simplify environment setup process
- Add automated dependency installation
- Create Docker containers for reproducible environments

---

## 13. Examples & Tutorials 📚

### Priority: LOW-MEDIUM
### Timeline: 1-2 weeks

#### 13.1 Comprehensive Examples
- Create step-by-step tutorials for common workflows
- Add Jupyter notebook examples
- Implement example projects with different architectures

#### 13.2 Best Practices Documentation
- Document architectural patterns and recommendations
- Add troubleshooting guides
- Create performance tuning guidelines

---

## Implementation Strategy

### Phase 1: Foundation (Weeks 1-4)
1. **Package structure reorganization** - Modular architecture
2. **Type safety and documentation** - Comprehensive type hints and docs
3. **Testing infrastructure** - Unit, integration, and performance tests
4. **Configuration management** - Unified configuration system

### Phase 2: Core Audio Features (Weeks 5-8)
1. **Processor architecture refactoring** - Clean separation and interfaces
2. **Loss function configuration system** - Flexible loss composition via YAML
3. **Callback-based logging system** - Modular, configurable logging
4. **Model factory patterns** - Easy model composition

### Phase 3: Advanced Features (Weeks 9-12)
1. **Data pipeline improvements** - Better data loading and preprocessing
2. **Performance optimization** - Memory and computation improvements
3. **Developer experience** - CLI, debugging tools, documentation
4. **Distribution and examples** - Packaging, tutorials, best practices

---

## Success Metrics

### Code Quality
- [ ] 90%+ test coverage
- [ ] All public APIs have type hints and documentation
- [ ] Zero critical code quality issues (linting, security)

### Developer Experience
- [ ] Setup time < 5 minutes
- [ ] Clear error messages with actionable guidance
- [ ] Comprehensive examples for all major features

### Performance
- [ ] Memory usage reduced by 20%
- [ ] Training speed improved by 15%
- [ ] Real-time inference capability maintained

### Maintainability
- [ ] Modular architecture with clear separation of concerns
- [ ] Configuration-driven development
- [ ] Comprehensive test suite with CI/CD

---

## Risk Mitigation

### Breaking Changes
- Implement deprecation warnings for old APIs
- Maintain backward compatibility for at least one major version
- Provide migration guides and automated migration tools

### Performance Regression
- Establish performance benchmarks before changes
- Implement performance regression testing
- Profile and optimize critical paths

### Community Impact
- Engage with users early for feedback
- Provide clear migration documentation
- Maintain stable release branches

---

## Long-term Vision

The goal is to transform NablAFx into a world-class, production-ready framework for neural audio effects modeling that:

1. **Scales** from research prototypes to production deployments
2. **Enables** rapid experimentation with clear architectural patterns
3. **Supports** the broader audio ML community with extensible interfaces
4. **Maintains** scientific rigor while improving usability
5. **Facilitates** reproducible research and fair comparisons

This roadmap positions NablAFx as the go-to framework for differentiable audio effects modeling, combining academic rigor with industrial-grade software engineering practices.
