# import sys
# import os
# from PyQt5.QtWidgets import QApplication

# # Import MVC components
# from mvc.models.thermometry_model import ThermometryModel
# from mvc.views.thermometry_view import ThermometryView
# from mvc.controllers.thermometry_controller import ThermometryController

# def main():
#     """Main application entry point."""
#     # Initialize Qt application
#     app = QApplication(sys.argv)
    
#     # Create MVC components
#     model = ThermometryModel()
#     view = ThermometryView()
#     controller = ThermometryController(model, view)
    
#     # Show the main window
#     view.show()
    
#     # Display welcome message
#     view.update_status("Welcome to MR Thermometry Application. Please load DICOM files to begin.")
    
#     # Start the application event loop
#     sys.exit(app.exec_())

# if __name__ == '__main__':
#     main()
# mvc/main.py

import sys
from PyQt5.QtWidgets import QApplication
from mvc.models.thermometry_model import ThermometryModel
from mvc.models.cinema_model import CinemaModel
from mvc.models.thermostat_model import ThermostatModel
from mvc.views.thermometry_view import ThermometryView
from mvc.controllers.thermometry_controller import ThermometryController
from mvc.controllers.advanced_features_controller import AdvancedFeaturesController

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Create models
    thermometry_model = ThermometryModel()
    cinema_model = CinemaModel(thermometry_model)
    thermostat_model = ThermostatModel()
    
    # Create view
    view = ThermometryView()
    
    # Create controllers
    thermometry_controller = ThermometryController(thermometry_model, view)
    advanced_features_controller = AdvancedFeaturesController(view, cinema_model, thermostat_model)
    
    # Show the main window
    view.show()
    
    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
