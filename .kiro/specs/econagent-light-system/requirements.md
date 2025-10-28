# Requirements Document

## Introduction

This document specifies the requirements for reproducing and improving the EconAgent paper system using Mesa for ABM simulation, LightAgent framework for agent architecture, and local LLM backends (NVIDIA Nemotron Docker + Ollama fallback) to ensure completely free operation. The system will simulate N=100 agents over 20 years (240 monthly steps) to reproduce macro indicators and analyses from the original ACL24-EconAgent paper while adding enterprise-level improvements including robustness, batching, caching, and comprehensive testing. All LLM calls will be local and free to avoid any API costs.

## Glossary

- **EconAgent_System**: The complete economic simulation system integrating Mesa, LightAgent, and local LLM backends
- **Mesa_Model**: The Mesa-based agent-based modeling framework managing the economic simulation
- **LightAgent_Framework**: The lightweight agentic framework (v0.4.0+) providing mem0 memory, Tools integration, and Tree-of-Thoughts (ToT) capabilities
- **Local_LLM_Client**: The HTTP client interface for communicating with local LLM services (Nemotron Docker + Ollama fallback)
- **Economic_Agent**: Individual agents in the simulation with perception, memory, and decision-making capabilities using LightAgent
- **Simulation_Environment**: The economic environment including government, bank, and goods market
- **Prompt_Templates**: Structured templates for agent perception, reflection, and decision-making prompts
- **DataCollector**: Mesa component for gathering simulation metrics and economic indicators
- **Batch_Processing**: System for processing multiple LLM requests efficiently using local resources
- **Memory_System**: Agent memory management using mem0 for storing dialogues and reflections
- **Docker_Nemotron**: Local NVIDIA Nemotron container running at localhost:8000
- **Ollama_Fallback**: Local Ollama service as backup LLM at localhost:11434

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to reproduce the original EconAgent paper results using completely local and free LLM setup, so that I can validate the methodology without any API costs or external dependencies.

#### Acceptance Criteria

1. WHEN the system initializes, THE EconAgent_System SHALL clone and integrate the original ACL24-EconAgent codebase
2. THE Local_LLM_Client SHALL connect to local NVIDIA Nemotron Docker container at http://localhost:8000/v1 using OpenAI-compatible format
3. WHEN Nemotron is unavailable, THE Local_LLM_Client SHALL fallback to Ollama service at http://localhost:11434/v1
4. THE EconAgent_System SHALL simulate 100 agents for 240 monthly steps by default with zero API costs
5. THE DataCollector SHALL reproduce key metrics from the paper including inflation, unemployment, GDP, Phillips curve, and Okun's law

### Requirement 2

**User Story:** As a developer, I want to integrate the production-level LightAgent framework (v0.4.0+) with economic agents, so that agents have enhanced memory, tools, and reasoning capabilities using the actual framework features.

#### Acceptance Criteria

1. THE LightAgent_Framework SHALL use mem0 memory stores for each Economic_Agent with configurable retention
2. WHEN an agent makes decisions, THE LightAgent_Framework SHALL use Tree-of-Thoughts reasoning with DeepSeek-R1 support if available locally
3. THE Memory_System SHALL store the last L=1 months of dialogues and quarterly reflections for each agent using mem0
4. WHEN a quarter ends, THE Economic_Agent SHALL perform reflection using LightAgent's self-learning capabilities
5. THE LightAgent_Framework SHALL expose economic tools for reading environment indicators including price, unemployment, and tax schedules

### Requirement 3

**User Story:** As a system administrator, I want enterprise-level robustness and performance features using only local resources, so that the system can handle large-scale simulations reliably without external costs.

#### Acceptance Criteria

1. THE Batch_Processing SHALL group multiple agent prompts into concurrent requests with configurable thread limits for local LLM services
2. THE EconAgent_System SHALL implement caching for identical prompts to reduce redundant local LLM calls
3. WHEN local LLM parsing fails, THE Local_LLM_Client SHALL implement automatic retry with fallback values and alternative local models
4. THE EconAgent_System SHALL validate all LLM outputs using JSON schema with value clamping to valid ranges
5. THE EconAgent_System SHALL support deterministic seeding for reproducible experiments using local resources only

### Requirement 4

**User Story:** As a researcher, I want comprehensive testing and validation capabilities using mock services, so that I can ensure system correctness and reliability without requiring actual LLM services.

#### Acceptance Criteria

1. THE EconAgent_System SHALL include unit tests for all major components including Local_LLM_Client, LightAgent_Framework integration, and Mesa_Model
2. THE EconAgent_System SHALL provide integration tests validating end-to-end simulation workflows with mock LLM responses
3. THE EconAgent_System SHALL include offline testing capabilities that simulate LLM responses without requiring running Docker containers
4. THE EconAgent_System SHALL validate that economic metrics remain within expected ranges during test simulations
5. THE EconAgent_System SHALL provide benchmarking capabilities for performance measurement using local resources

### Requirement 5

**User Story:** As a user, I want clear setup and execution instructions for local-only deployment, so that I can run the system locally with minimal configuration and zero ongoing costs.

#### Acceptance Criteria

1. THE EconAgent_System SHALL provide automated scripts for starting local NVIDIA Nemotron Docker container and Ollama service
2. THE EconAgent_System SHALL include a CLI interface with configurable parameters for agents, years, batch size, and seeding
3. THE EconAgent_System SHALL provide comprehensive documentation including local setup, Docker configuration, and troubleshooting guides
4. THE EconAgent_System SHALL validate local service connectivity and provide clear error messages for missing Docker or Ollama dependencies
5. THE EconAgent_System SHALL support both full simulations and quick debugging runs with reduced parameters using local resources

### Requirement 6

**User Story:** As an economic modeler, I want accurate agent decision-making based on economic context using LightAgent's production features, so that the simulation produces realistic economic behaviors with advanced agent capabilities.

#### Acceptance Criteria

1. THE Prompt_Templates SHALL implement perception prompts including economic context, personal history, and market conditions using LightAgent's prompt system
2. THE Economic_Agent SHALL make work propensity decisions as float values in [0,1] with 0.02 step precision using LightAgent decision framework
3. THE Economic_Agent SHALL make consumption decisions as float values in [0,1] representing fraction of available assets
4. THE Economic_Agent SHALL consider savings, expected income, prices, interest rates, taxes, and redistribution in decisions using LightAgent tools
5. THE Economic_Agent SHALL adapt decision-making based on quarterly reflections and learned patterns using LightAgent's self-learning capabilities

### Requirement 7

**User Story:** As a simulation operator, I want robust economic environment modeling integrated with LightAgent's tool system, so that the system accurately represents government, banking, and market dynamics with intelligent agent interactions.

#### Acceptance Criteria

1. THE Simulation_Environment SHALL implement progressive taxation using 2018 U.S. Federal tax brackets accessible via LightAgent tools
2. THE Simulation_Environment SHALL distribute tax revenue equally across all agents monthly with LightAgent tool integration
3. THE Simulation_Environment SHALL update wages and prices using equations (7) and (8) from the original paper
4. THE Simulation_Environment SHALL apply annual interest on savings and update interest rates via Taylor rule
5. THE Simulation_Environment SHALL maintain inventory tracking for goods production and consumption accessible through LightAgent economic tools

### Requirement 8

**User Story:** As a performance engineer, I want optimized local LLM usage and resource management, so that large-scale simulations complete in reasonable time using only local computational resources.

#### Acceptance Criteria

1. THE Batch_Processing SHALL support configurable parallel thread limits for concurrent local LLM requests
2. THE EconAgent_System SHALL implement in-memory caching for identical prompt-response pairs to minimize local LLM calls
3. THE Local_LLM_Client SHALL support streaming responses when beneficial for performance with local services
4. THE EconAgent_System SHALL provide graceful degradation when local LLM services are unavailable with conservative fallback decisions
5. THE EconAgent_System SHALL include profiling capabilities for identifying performance bottlenecks in local LLM processing

### Requirement 9

**User Story:** As a cost-conscious researcher, I want to ensure zero ongoing operational costs, so that I can run extensive experiments without financial constraints.

#### Acceptance Criteria

1. THE EconAgent_System SHALL operate entirely with local computational resources without external API calls
2. THE Local_LLM_Client SHALL never make requests to paid external services or APIs
3. THE EconAgent_System SHALL provide clear documentation on local resource requirements and setup costs (one-time hardware/software)
4. THE EconAgent_System SHALL include monitoring to detect and prevent any accidental external API calls
5. THE EconAgent_System SHALL provide performance estimates for different local hardware configurations to help users plan resources