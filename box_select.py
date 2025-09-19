import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import cv2
import numpy as np

class MatplotlibCoordinateSelector:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.start_point = None
        self.current_rect = None
        self.regions = {}
        self.current_field = ""
        self.drawn_patches = []  # Keep track of drawn rectangles and text
        
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.img_rgb)
        self.ax.set_title("Click and drag to select regions. Press 'n' for next field, 'q' to quit.")
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Field names to capture
        self.field_names = [
            'invoice', 'date', 'tech',
            'acct_name', 'acct_phone', 'acct_address', 'acct_CSZ', 'acct_email',
            'project_name', 'project_phone', 'project_address', 
            'work_line_1', 'work_line_2', 'work_line_3', 'work_line_4', 'work_line_5', 'work_line_6', 'work_line_7', 'work_line_8', 'work_line_9', 'work_line_10', 'work_line_11',
            'work_price_1', 'work_price_2', 'work_price_3', 'work_price_4', 'work_price_5', 'work_price_6', 'work_price_7', 'work_price_8', 'work_price_9', 'work_price_10', 'work_price_11',
            'check_num', 'total_amount'
        ]
        self.current_field_index = 0
        self.update_title()
    
    def update_title(self):
        if self.current_field_index < len(self.field_names):
            field_name = self.field_names[self.current_field_index]
            self.ax.set_title(f"Select region for: {field_name} (Press 'n' for next, 'u' to undo, 'q' to quit)")
        else:
            self.ax.set_title("All fields captured! Press 'q' to quit and see results.")
    
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.start_point = (int(event.xdata), int(event.ydata))
        print(f"Start point: {self.start_point}")
    
    def on_release(self, event):
        if event.inaxes != self.ax or self.start_point is None:
            return
            
        if self.current_field_index >= len(self.field_names):
            return
            
        end_point = (int(event.xdata), int(event.ydata))
        field_name = self.field_names[self.current_field_index]
        
        # Calculate rectangle coordinates (x1, y1, x2, y2)
        x1 = min(self.start_point[0], end_point[0])
        y1 = min(self.start_point[1], end_point[1])
        x2 = max(self.start_point[0], end_point[0])
        y2 = max(self.start_point[1], end_point[1])
        
        # Store the region
        self.regions[field_name] = (x1, y1, x2, y2)
        
        # Draw rectangle on the image
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
        self.ax.add_patch(rect)
        
        # Add text label
        text_obj = self.ax.text(x1, y1-5, field_name, fontsize=8, color='red', weight='bold')
        
        # Store the visual elements for potential undo
        self.drawn_patches.append((rect, text_obj))
        
        print(f"'{field_name}': ({x1}, {y1}, {x2}, {y2}),")
        
        # Move to next field
        self.current_field_index += 1
        self.update_title()
        
        # Refresh the plot
        self.fig.canvas.draw()
        
        self.start_point = None
    
    def on_key_press(self, event):
        if event.key == 'n':  # Next field (skip current)
            if self.current_field_index < len(self.field_names):
                print(f"Skipped: {self.field_names[self.current_field_index]}")
                self.current_field_index += 1
                self.update_title()
                self.fig.canvas.draw()
        
        elif event.key == 'u':  # Undo last selection
            self.undo_last_selection()
        
        elif event.key == 'q':  # Quit
            self.print_results()
            plt.close()
    
    def undo_last_selection(self):
        """Undo the last field selection"""
        if self.current_field_index > 0:
            # Move back to previous field
            self.current_field_index -= 1
            field_name = self.field_names[self.current_field_index]
            
            # Remove from regions dict if it exists
            if field_name in self.regions:
                del self.regions[field_name]
                print(f"Undoing: {field_name}")
                
                # Remove the visual elements (rectangle and text)
                if self.drawn_patches:
                    rect, text_obj = self.drawn_patches.pop()
                    rect.remove()
                    text_obj.remove()
                
                # Update title and refresh
                self.update_title()
                self.fig.canvas.draw()
            else:
                print(f"Repositioning to: {field_name}")
                self.update_title()
        else:
            print("Nothing to undo - at the first field")
    
    def print_results(self):
        print("\n" + "="*50)
        print("COPY THIS INTO YOUR CODE:")
        print("="*50)
        print("regions = {")
        for field_name, coords in self.regions.items():
            print(f"    '{field_name}': {coords},")
        print("}")
        print("="*50)
    
    def select_regions(self):
        plt.show()

# Usage
if __name__ == "__main__":
    selector = MatplotlibCoordinateSelector('form_img.jpeg')
    selector.select_regions()