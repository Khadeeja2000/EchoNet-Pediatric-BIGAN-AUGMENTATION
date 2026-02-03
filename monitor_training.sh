#!/bin/bash
# Monitor GPU and system resources during training

echo "=========================================="
echo "  Training Monitor"
echo "=========================================="
echo ""
echo "Monitoring system resources..."
echo "Press Ctrl+C to stop"
echo ""

# Monitor in loop
while true; do
    clear
    echo "=========================================="
    echo "  BiGAN Training Monitor"
    echo "=========================================="
    date
    echo ""
    
    # CPU usage
    echo "CPU Usage:"
    top -l 1 | grep "CPU usage" | head -1
    echo ""
    
    # Memory usage
    echo "Memory Usage:"
    top -l 1 | grep "PhysMem" | head -1
    echo ""
    
    # Check if training process is running
    if pgrep -f "train_bigan_improved.py" > /dev/null; then
        echo "✅ Training is RUNNING"
        echo ""
        
        # Show training process details
        ps aux | grep "train_bigan_improved.py" | grep -v grep | head -1 | awk '{printf "  PID: %s\n  CPU: %s%%\n  Memory: %s%%\n", $2, $3, $4}'
        echo ""
        
        # Show latest training logs if available
        if [ -f "training.log" ]; then
            echo "Latest training output:"
            tail -5 training.log | sed 's/^/  /'
        fi
    else
        echo "⏸️  Training is NOT running"
    fi
    
    echo ""
    echo "=========================================="
    echo "💡 Open Activity Monitor > Window > GPU History"
    echo "   to see GPU usage in real-time"
    echo "=========================================="
    
    sleep 5
done

