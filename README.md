## 🔍 Real-time Anomaly Detection System (PPO)

<h3 align="center">TEAM ChocoPytorch</h1>

### 📋 Table of Contents

- [Project Introduction](#project-introduction)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage Guide](#usage-guide)
- [Performance Metrics](#performance-metrics)
- [Contributors](#contributors)

### 🎯 Project Introduction

This project develops an anomaly detection and prediction model for time-series data collected in real-time from manufacturing process equipment.

**Key Features:**
- 🤖 **PPO (Proximal Policy Optimization)** based reinforcement learning model
- 📊 **Real-time data visualization** and monitoring
- ⚡ **Sequential data processing** (sequential processing from index 0)
- 🎛️ **Manual/Automatic update** mode support
- 📈 **Real-time performance metrics** monitoring

### ✨ Key Features

#### 1. Real-time Anomaly Detection
- Real-time anomaly detection using PPO reinforcement learning model
- Stable detection performance through sequential data processing
- Real-time chart updates visualizing current processing status

#### 2. Interactive Web Interface
- Intuitive user interface based on Streamlit
- Real-time charts and performance metrics display
- Easy control with start/stop/reset buttons

#### 3. Performance Monitoring
- Real-time calculation of accuracy, precision, recall, and F1 score
- Anomaly detection list and detailed information display
- Processing progress and current status monitoring

#### 4. Flexible Update Modes
- **Automatic Update**: Automatic progression with configurable intervals
- **Manual Update**: Step-by-step progression with button clicks (solves scrolling issues)

### 🛠️ Technology Stack

#### Backend & Algorithm
- **Python 3.8+**
- **PyTorch**: PPO model implementation
- **Streamlit**: Web application framework
- **Plotly**: Real-time data visualization
- **Pandas**: Data processing and analysis
- **NumPy**: Numerical computation

#### Machine Learning
- **PPO (Proximal Policy Optimization)**: Reinforcement learning-based anomaly detection
- **Custom Environment**: Evaluation environment for anomaly detection
- **Real-time Processing**: Sequential data processing

#### Data Visualization
- **Plotly Graph Objects**: Real-time chart generation
- **Interactive Charts**: Current processing points and anomaly detection display
- **Responsive Design**: Support for various screen sizes

### 📁 Project Structure

```
bistelligence/
├── meta_aad/                       # Reinforcement learning model
│   ├── env.py                      # Environment definition
│   ├── ppo2.py                     # PPO model implementation
│   ├── agents.py                   # Agent definition
│   └── utils.py                    # Utility functions
│
├── data/                           # Data files
│   └── sensor_data_with_anomalylabel_isolationforest.csv
│
├── log/                            # Training logs
│   └── model.pth                   # Trained PPO model
│
├── results/                        # Result files
│
├── src/                            # Source code
│   └── img/                        # Image files
│
├── util/                           # Utilities
│   ├── eda.py                      # Exploratory data analysis
│   └── preprocess.py               # Data preprocessing
│
├── app.py                          # Main Streamlit application
├── evaluate.py                     # Model evaluation script
├── train.py                        # Model training script
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

### 🚀 Installation and Setup

#### 1. Clone Repository
```bash
git clone https://github.com/ChocoPytorch/BISTelligence.git
cd BISTelligence
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run Application
```bash
streamlit run app.py
```

#### 4. Access in Browser
```
http://localhost:8501
```

### 📖 Usage Guide

#### 1. Application Startup
- Access `http://localhost:8501` in browser
- Confirm data loading and model initialization completion

#### 2. Configuration Adjustment
- **Automatic Update**: Select automatic/manual mode with checkbox
- **Update Interval**: Set processing interval in automatic mode (0.1~2.0 seconds)
- **Chart Window Size**: Number of data points to display in chart (50~200)

#### 3. Start Anomaly Detection
- **▶️ Start**: Start real-time anomaly detection
- **⏸️ Stop**: Pause detection
- **🔄 Reset**: Reset all states

#### 4. View Results
- **Real-time Chart**: Display current processing data points
- **Performance Metrics**: Accuracy, precision, recall, F1 score
- **Anomaly Detection List**: Detailed information of detected anomalies

### 📊 Performance Metrics

#### Real-time Monitoring
- **Accuracy**: Ratio of correct predictions among total predictions
- **Precision**: Ratio of actual anomalies among detected anomalies
- **Recall**: Ratio of detected anomalies among actual anomalies
- **F1 Score**: Harmonic mean of precision and recall

#### Visualization
- **Real-time Chart**: Highlight current processing data points
- **Anomaly Detection Display**: Mark detected anomaly points with red X
- **Progress Rate**: Processing completion ratio compared to total data

### 🔧 Major Improvements

#### Recent Updates
- ✅ **Sequential Data Processing**: Sequential processing from index 0
- ✅ **Real-time Chart Enhancement**: Visualization of current processing points
- ✅ **Scroll Issue Resolution**: Added manual update mode
- ✅ **Performance Metrics Display**: Real-time performance monitoring
- ✅ **Anomaly Detection List**: Separate display of detected anomalies only

#### Technical Improvements
- **PPO Model Integration**: Reinforcement learning-based anomaly detection
- **Environment-based Evaluation**: Accurate evaluation using `EvalEnv`
- **Enhanced Exception Handling**: Stable application execution
- **UI/UX Improvements**: Intuitive user interface

### 👥 Contributors

Team members who participated in this project.

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Chocopytorch"><img src="https://avatars.githubusercontent.com/u/122209595?v=4?s=100" width="100px;" alt="Chocopytorch"/><br /><sub><b>Chocopytorch</b></sub></a><br /><a href="https://github.com/Chocopytorch/BISTelligence/commits?author=Chocopytorch" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chosungsu"><img src="https://avatars.githubusercontent.com/u/48382347?v=4?s=100" width="100px;" alt="chosungsu"/><br /><sub><b>chosungsu</b></sub></a><br /><a href="https://github.com/Chocopytorch/BISTelligence/commits?author=chosungsu" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kmw4097"><img src="https://avatars.githubusercontent.com/u/98750892?v=4?s=100" width="100px;" alt="kmw4097"/><br /><sub><b>kmw4097</b></sub></a><br /><a href="https://github.com/Chocopytorch/BISTelligence/commits?author=kmw4097" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dbnub"><img src="https://avatars.githubusercontent.com/u/99518647?v=4?s=100" width="100px;" alt="dbnub"/><br /><sub><b>dbnub</b></sub></a><br /><a href="https://github.com/Chocopytorch/BISTelligence/commits?author=dbnub" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/choiyongwoo"><img src="https://avatars.githubusercontent.com/u/50268222?v=4?s=100" width="100px;" alt="choiyongwoo"/><br /><sub><b>choiyongwoo</b></sub></a><br /><a href="https://github.com/Chocopytorch/BISTelligence/commits?author=choiyongwoo" title="Commits">📖</a> </td>
    </tr>
  </tbody>
</table>

---

**BISTelligence.ai** - Manufacturing Process Anomaly Detection Solution

