# Email Analyser

## Overview
Email Analyser is a Python-based application that processes and analyses email content. It extracts useful information such as sender interaction patterns, sentiment analysis, topics of interest, and technology discussions. The application is built using Streamlit and runs inside a Docker container for ease of deployment.

## Features
- Extracts email metadata including sender, recipients, and timestamps
- Analyzes sender interaction patterns
- Performs sentiment analysis on email content
- Identifies topics of interest and technology discussions
- Provides visual insights via a Streamlit dashboard
- Export results as CSV

## Prerequisites
Ensure you have the following installed:
- Docker
- Git (optional, for cloning the repository)

## Installation
### 1. Clone the Repository (Optional)
If using Git:
```sh
git clone -b develop https://github.com/vishnu-pits/email_content_challenge.git
cd email_content_challenge
```

### 2. Build the Docker Image
Run the following command to build the Docker image:
```sh
docker-compose build
```

### 3. Run the Application
Start the container and expose the necessary port:
```sh
docker-compose up -d
```
This will launch the Streamlit application and make it accessible at:
```
http://localhost:8501
```

## License
This project is licensed under the MIT License.

