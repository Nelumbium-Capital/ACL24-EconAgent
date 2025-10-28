# Implementation Plan

## Overview

This implementation plan converts the EconAgent-Light design into a series of coding tasks that build incrementally on the original ACL24-EconAgent codebase. Each task focuses on specific code implementation while preserving the economic logic and migrating to Mesa + LightAgent + local LLM architecture.

## Implementation Tasks

- [x] 1. Project Setup and Original Code Integration
  - Clone and analyze the original ACL24-EconAgent repository structure
  - Create new project structure integrating original economic logic with Mesa/LightAgent
  - Setup development environment with Mesa, LightAgent, and local LLM dependencies
  - Extract and document original economic parameters and equations from config.yaml and simulate.py
  - _Requirements: 1.1, 5.1, 5.2_

- [ ] 2. Local LLM Client Implementation
  - [x] 2.1 Implement NemotronClient for local Docker communication
    - Create HTTP client for NVIDIA Nemotron Docker container at localhost:8000
    - Implement OpenAI-compatible request/response format
    - Add error handling, retries, and connection validation
    - _Requirements: 1.2, 3.3, 5.4_
  
  - [x] 2.2 Add Ollama fallback client
    - Implement Ollama client as backup LLM service at localhost:11434
    - Create unified interface for switching between Nemotron and Ollama
    - Add service health checking and automatic fallback logic
    - _Requirements: 1.3, 8.4_
  
  - [x] 2.3 Implement response validation and caching
    - Create JSON schema validation for agent decisions (work/consumption in [0,1] with 0.02 steps)
    - Implement value clamping and normalization functions
    - Add in-memory caching for identical prompts to reduce LLM calls
    - Create fallback decision logic when LLM fails (work=0.2, consumption=0.1)
    - _Requirements: 3.4, 8.2_

- [ ] 3. LightAgent Integration Layer
  - [x] 3.1 Create LightAgentWrapper class
    - Initialize LightAgent with mem0 memory system
    - Configure Tree-of-Thought reasoning for economic decisions
    - Setup economic tools for environment data access
    - Integrate with local LLM clients (Nemotron/Ollama)
    - _Requirements: 2.1, 2.2, 2.5_
  
  - [x] 3.2 Implement economic decision-making interface
    - Create decide() method that takes agent profile and environment snapshot
    - Build economic context prompts using original simulate.py prompt logic
    - Process LLM responses and return validated work/consumption decisions
    - Handle quarterly reflection prompts and memory updates
    - _Requirements: 6.1, 6.4, 6.5_
  
  - [x] 3.3 Setup memory and learning system
    - Configure mem0 for storing agent dialogues and reflections (L=1 month window)
    - Implement quarterly reflection mechanism using LightAgent self-learning
    - Create memory retrieval for decision-making context
    - Add learning pattern tracking and behavioral adaptation
    - _Requirements: 2.3, 2.4_

- [ ] 4. Mesa Model Implementation
  - [x] 4.1 Create EconModel class extending Mesa.Model
    - Migrate original config.yaml parameters to Mesa model initialization
    - Setup Mesa RandomActivationByType scheduler for agent coordination
    - Initialize economic environment variables (price, wages, interest rates, taxes)
    - Create DataCollector for tracking economic indicators
    - _Requirements: 1.1, 1.4, 7.3, 7.4, 7.5_
  
  - [x] 4.2 Implement economic environment dynamics
    - Port original price and wage update equations from simulate.py
    - Implement progressive taxation using 2018 U.S. Federal tax brackets
    - Add tax collection and redistribution logic
    - Create interest rate updates using Taylor rule
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [x] 4.3 Create model step() method
    - Implement monthly simulation step replicating original simulate.py logic
    - Coordinate agent decision-making phase
    - Execute economic calculations (production, consumption, market updates)
    - Trigger quarterly reflections every 3 months
    - Collect and store economic metrics
    - _Requirements: 1.4, 6.3, 7.5_

- [ ] 5. Economic Agent Implementation
  - [x] 5.1 Create EconAgent class extending Mesa.Agent
    - Port original agent attributes from ai-economist BasicMobileAgent
    - Initialize agent profile (skill, wealth, job, demographics) using original distributions
    - Integrate LightAgentWrapper for intelligent decision-making
    - Setup agent memory system and learning capabilities
    - _Requirements: 2.1, 6.2, 6.3_
  
  - [x] 5.2 Implement agent step() method
    - Build economic context snapshot for LightAgent decision-making
    - Call LightAgent for work propensity and consumption decisions
    - Apply decisions using original economic formulas
    - Update agent financial state (income, savings, taxes, consumption)
    - _Requirements: 6.2, 6.3, 6.4_
  
  - [x] 5.3 Add quarterly reflection mechanism
    - Implement reflect() method called every 3 months by model
    - Use LightAgent to process quarterly economic data and generate insights
    - Update agent memory and learning patterns
    - Store reflection results for future decision-making
    - _Requirements: 2.4, 6.5_

- [ ] 6. Prompt Templates and Economic Tools
  - [x] 6.1 Create economic prompt templates
    - Port original problem_prompt, job_prompt, and economic context from simulate.py
    - Implement perception prompts with economic indicators and personal history
    - Create quarterly reflection prompts for learning and adaptation
    - Add prompt formatting and prettification functions
    - _Requirements: 6.1, 6.4_
  
  - [x] 6.2 Implement economic tools for LightAgent
    - Create tools for accessing current economic indicators (price, unemployment, inflation)
    - Add tax bracket and rate lookup tools
    - Implement market condition analysis tools
    - Create personal financial history access tools
    - _Requirements: 2.5, 7.1, 7.2_

- [ ] 7. Batch Processing and Performance Optimization
  - [ ] 7.1 Implement concurrent agent processing
    - Create batch processing for multiple agent LLM requests
    - Add configurable thread pool for parallel processing
    - Implement request queuing and response aggregation
    - Add error handling for batch failures with individual fallbacks
    - _Requirements: 3.1, 8.1_
  
  - [ ] 7.2 Add performance monitoring and optimization
    - Implement caching hit rate monitoring
    - Add LLM request timing and success rate tracking
    - Create performance profiling for bottleneck identification
    - Add resource usage monitoring for local LLM services
    - _Requirements: 8.5_

- [ ] 8. Data Collection and Visualization
  - [x] 8.1 Implement comprehensive data collection
    - Create DataCollector configuration for all economic indicators
    - Track inflation, unemployment, GDP, wage levels, and tax revenue
    - Collect agent-level data (decisions, wealth, consumption patterns)
    - Store simulation results in structured format for analysis
    - _Requirements: 1.5, 4.4_
  
  - [x] 8.2 Create visualization and analysis tools
    - Implement plotting functions for key economic indicators over time
    - Create Phillips curve and Okun's law analysis (replicating original paper figures)
    - Add agent behavior analysis and decision pattern visualization
    - Generate comparison reports between original and new implementation
    - _Requirements: 1.5_

- [ ] 9. CLI Interface and Configuration
  - [x] 9.1 Create command-line interface
    - Implement CLI with configurable parameters (agents, years, batch size, seeding)
    - Add local LLM service configuration options
    - Create quick debugging mode with reduced parameters
    - Add verbose logging and progress reporting
    - _Requirements: 5.2, 5.3_
  
  - [x] 9.2 Add Docker and service management scripts
    - Create automated scripts for starting Nemotron Docker container
    - Add Ollama service setup and configuration scripts
    - Implement service health checking and startup validation
    - Create environment setup and dependency installation scripts
    - _Requirements: 5.1, 5.4_

- [ ] 10. Testing and Validation
  - [ ] 10.1 Create unit tests for core components
    - Test local LLM clients with mock responses
    - Validate economic calculations against original formulas
    - Test LightAgent integration and memory systems
    - Verify Mesa model step execution and data collection
    - _Requirements: 4.1, 4.4_
  
  - [ ] 10.2 Implement integration tests
    - Create end-to-end simulation tests with small agent populations (N=10)
    - Validate economic indicators remain within expected ranges
    - Test batch processing and concurrent agent execution
    - Verify memory persistence and learning mechanisms
    - _Requirements: 4.2, 4.4_
  
  - [ ]* 10.3 Add performance benchmarking
    - Create benchmarks comparing original vs new implementation performance
    - Test scalability with different agent population sizes
    - Measure LLM request efficiency and caching effectiveness
    - Validate resource usage under different hardware configurations
    - _Requirements: 4.5, 8.1, 8.5_

- [ ] 11. Documentation and Deployment
  - [x] 11.1 Create comprehensive documentation
    - Write detailed setup instructions for local LLM services
    - Document migration from original codebase and key differences
    - Create user guide for running simulations and interpreting results
    - Add troubleshooting guide for common issues
    - _Requirements: 5.3, 5.5_
  
  - [ ] 11.2 Add reproducibility and validation
    - Create scripts to reproduce original paper results
    - Add deterministic seeding for experiment reproducibility
    - Implement result validation against original simulation outputs
    - Create example notebooks demonstrating key features
    - _Requirements: 3.5, 9.3_

- [ ]* 12. Advanced Features and Optimizations
  - [ ]* 12.1 Implement advanced LightAgent features
    - Add Tree-of-Thought reasoning with multiple candidate evaluation
    - Implement advanced memory management with configurable retention policies
    - Create agent collaboration mechanisms for market information sharing
    - Add adaptive tool selection based on economic conditions
    - _Requirements: 2.2_
  
  - [ ]* 12.2 Add experimental economic extensions
    - Implement additional economic scenarios beyond the original paper
    - Add support for different tax policies and economic interventions
    - Create agent heterogeneity experiments with different behavioral patterns
    - Add economic shock simulation capabilities
    - _Requirements: 7.1, 7.2_