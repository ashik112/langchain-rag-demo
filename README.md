# LangChain RAG Demo

A demo application that showcases the implementation of RAG (Retrieval-Augmented Generation) using LangChain.

## Prerequisites

- Docker version 28.0.0 or higher
- Docker Compose version 2.26.0 or higher
- Python 3.11
- Google API Key for Gemini AI

## Environment Setup

1. Create a `.env` file in the project root directory:
```bash
cp .env.sample .env
```

2. Edit the `.env` file and add your Google API Key:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

## Running with Docker

### Build and Run

```bash
docker compose up --build
```

### Build Only

```bash
docker compose build
```

### Run Only

```bash
docker compose up
```

### Stop Containers

```bash
docker compose down
```

### Access the Application

Once the containers are running, you can access the application at:
- http://localhost:5000

## Docker Configuration Details

### Dockerfile

The Dockerfile uses a multi-stage build approach:
1. Build stage (`builder`): Installs dependencies and builds the application
2. Final stage (`app`): Creates a minimal production image

### docker-compose.yml

The docker-compose.yml file is configured with:
- Port mapping: 5000:5000
- Environment variables from .env file
- Volume mounts for persistent storage
- Resource limits and reservations
- Healthcheck configuration

## Project Structure

```
langchain-rag-demo/
├── assets/           # Static assets
├── faiss_index/      # FAISS index storage
├── web/              # Frontend code
├── .env              # Environment variables
├── .gitignore        # Git ignore rules
├── Dockerfile        # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── main.py           # Main application file
├── requirements.txt  # Python dependencies
└── web_server.py     # Web server implementation
```

## Troubleshooting

### Common Issues

1. **Missing Google API Key**
   - Ensure GOOGLE_API_KEY is set in your .env file
   - Get your API key from https://g.co/ai/idxGetGeminiKey

2. **Docker Build Errors**
   - Make sure you're using Docker 28.0.0 or higher
   - Ensure all required system dependencies are installed

3. **Port Conflicts**
   - If port 5000 is already in use, stop the conflicting service or modify the port mapping in docker-compose.yml

## Security Notes

- Never commit your .env file with real API keys
- The .env file is already included in .gitignore
- Use environment variables for sensitive information

## License

This project is licensed under the MIT License - see the LICENSE file for details.
