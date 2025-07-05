#!/bin/bash
# Development server (deprecated - use ./start.sh instead)

echo "⚠️  This script is deprecated. Please use:"
echo "   ENV=development ./start.sh"
echo ""
echo "Or simply:"
echo "   ./start.sh"
echo ""
echo "Starting development server anyway..."

export ENV=development
./start.sh