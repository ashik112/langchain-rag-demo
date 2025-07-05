# LangChain RAG Demo

A simple, production-ready RAG system that works anywhere with minimal configuration.

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone <your-repo>
   cd langchain-rag-demo
   ```

2. **Configure (copy and edit)**
   ```bash
   cp config.env .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

3. **Add Your Documents**
   - Place PDF, DOCX, or TXT files in the `assets/` directory

4. **Start the Application**
   ```bash
   ./start.sh
   ```

That's it! Open `http://localhost:5000` and start asking questions.

## 🎯 Simple Commands

```bash
# Development (default)
./start.sh

# Production
ENV=production ./start.sh

# Small servers (e2-micro, etc.)
ENV=micro ./start.sh

# Docker
ENV=docker ./start.sh
```

## 🔧 Configuration

Edit your `.env` file:

```bash
# Required
GOOGLE_API_KEY=your_key_here

# Environment: development, production, micro, docker
ENV=development

# Port
PORT=5000

# For small servers, use:
# ENV=micro
# CPU_LIMIT=0.8
# MEMORY_LIMIT=800M
```

## 🌐 Deploy Anywhere

The system adapts automatically:

- **Development**: Single worker, debug mode, hot reload
- **Production**: Auto-scaling workers, optimized performance
- **Micro**: 2 workers, reduced memory usage (perfect for e2-micro)
- **Docker**: Containerized with proper resource limits

## 📁 File Structure

```
├── main.py              # Flask application
├── rag_system.py        # RAG logic
├── wsgi.py              # WSGI entry point
├── start.sh             # Universal start script
├── gunicorn.conf.py     # Auto-configuring Gunicorn
├── docker-compose.yml   # Environment-adaptive Docker
├── config.env           # Configuration template
└── assets/              # Your documents go here
```

## 🔄 Migration

To move to a new server:

1. Copy your files
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables in `.env`
4. Run: `./start.sh`

## 🐳 Docker Deployment

```bash
# Copy config and start
cp config.env .env
# Edit .env with your settings
docker-compose up --build -d
```

## 🛠️ Development

```bash
# Auto-reload development server
ENV=development ./start.sh

# Or traditional Flask
python -m flask --app main run --debug
```

## 📊 Monitoring

```bash
# Check status
curl http://localhost:5000/api/sessions-info

# View logs
docker-compose logs -f  # Docker
tail -f logs/app.log     # Local
```

## 🔧 Troubleshooting

- **Out of memory**: Use `ENV=micro`
- **Slow responses**: Increase timeout in `gunicorn.conf.py`
- **Port conflicts**: Change `PORT=8080` in `.env`

Simple, adaptable, and ready for production! 🎉
