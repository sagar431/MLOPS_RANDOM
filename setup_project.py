import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/interim',
        'data/processed',
        'models',  # DVC will track this directory
        'reports',
        'logs'
    ]
    
    # Create .gitkeep files to ensure git tracks empty directories
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            # Create .gitkeep file in each directory
            with open(os.path.join(directory, '.gitkeep'), 'w') as f:
                pass
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {str(e)}")

if __name__ == "__main__":
    logger.info("Setting up project directories...")
    create_directories()
    logger.info("Project setup completed!") 