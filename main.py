import os
import sys

if __name__ == '__main__':
    print("Starting F1 Analytics Dashboard...")
    from dashboard.app import app
    app.run(debug=True, port=8050)
