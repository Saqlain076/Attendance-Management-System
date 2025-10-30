import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
import cv2
from PIL import Image, ImageTk, ImageEnhance
import face_recognition
import mysql.connector
from datetime import datetime
import os
import winsound
from dotenv import load_dotenv
import pytesseract
import numpy as np
import pickle

load_dotenv()


pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class ModernButton(ttk.Button):
    def _init_(self, master=None, **kwargs):
        super()._init_(master, **kwargs)
        self.configure(style='Modern.TButton')

def augment_image(image):
    """Apply random augmentations to the image"""
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Brightness variations
    for factor in [0.8, 1.2]:
        enhancer = ImageEnhance.Brightness(image)
        augmented_images.append(enhancer.enhance(factor))
        
    # Contrast variations
    for factor in [0.8, 1.2]:
        enhancer = ImageEnhance.Contrast(image)
        augmented_images.append(enhancer.enhance(factor))
    
    # Slight rotations
    for angle in [-10, 10]:
        augmented_images.append(image.rotate(angle, expand=True))
    
    # Horizontal flip
    augmented_images.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    
    return augmented_images

class AttendanceSystem:
    def __init__(self, root):

        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("1400x900")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('Modern.TButton', 
                           font=('Helvetica', 12, 'bold'),
                           padding=10)
        self.style.configure('Title.TLabel',
                           font=('Helvetica', 24, 'bold'),
                           padding=10)
        self.style.configure('Header.TLabel',
                           font=('Helvetica', 14, 'bold'),
                           padding=5)
        self.style.configure('Status.TLabel',
                           font=('Helvetica', 12),
                           padding=5)
        
        # Initialize variables
        self.camera = None
        self.is_camera_on = False
        self.known_face_encodings = []
        self.known_face_names = []
        self.current_mode = None
        self.frame_count = 0
        self.process_every_n_frames = 3  # Process every 3rd frame
        self.last_processed_frame = None
        self.valid_registration_numbers = set()  # Store valid registration numbers
        
        # Load valid registration numbers
        self.load_valid_registration_numbers()
        
        # Database connection
        self.db_connection = self.create_db_connection()
        
        # Load known faces
        self.load_known_faces()
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        # Main container
        self.main_container = ttk.Frame(self.root, padding="2")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_frame = ttk.Frame(self.main_container)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, 
                              text="Smart Attendance System",
                              style='Title.TLabel')
        title_label.pack()
        
        # Mode selection frame
        mode_frame = ttk.LabelFrame(self.main_container, 
                                  text="Recognition Mode",
                                  padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Mode buttons with icons
        self.face_btn = ModernButton(mode_frame,
                                   text="Face Recognition",
                                   command=lambda: self.start_recognition("face"))
        self.face_btn.pack(side=tk.LEFT, padx=10, expand=True)
        
        self.card_btn = ModernButton(mode_frame,
                                   text="Card Recognition",
                                   command=lambda: self.start_recognition("card"))
        self.card_btn.pack(side=tk.LEFT, padx=10, expand=True)
        
        # Main content area
        content_frame = ttk.Frame(self.main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for camera
        left_frame = ttk.LabelFrame(content_frame, text="Camera Feed", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera feed
        self.camera_frame = ttk.Frame(left_frame)
        self.camera_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Status display
        self.status_label = ttk.Label(left_frame,
                                    text="Select a recognition mode to begin",
                                    style='Status.TLabel')
        self.status_label.pack(pady=10)
        
        # Right frame for attendance table
        right_frame = ttk.LabelFrame(content_frame, text="Attendance Records", padding="10")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create Treeview with modern style
        self.tree = ttk.Treeview(right_frame,
                                columns=("Registration No", "Date", "Time In", "Time Out"),
                                show="headings",
                                height=20)
        
        # Configure Treeview style
        self.style.configure("Treeview",
                           font=('Helvetica', 10),
                           rowheight=30)
        self.style.configure("Treeview.Heading",
                           font=('Helvetica', 10, 'bold'))
        
        # Set column headings and widths
        self.tree.heading("Registration No", text="Registration No")
        self.tree.heading("Date", text="Date")
        self.tree.heading("Time In", text="Time In")
        self.tree.heading("Time Out", text="Time Out")
        
        self.tree.column("Registration No", width=150, anchor=tk.CENTER)
        self.tree.column("Date", width=100, anchor=tk.CENTER)
        self.tree.column("Time In", width=100, anchor=tk.CENTER)
        self.tree.column("Time Out", width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the tree and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load existing attendance records
        self.load_attendance_records()
        
        # Configure grid weights
        self.main_container.pack_propagate(False)
        
    def create_db_connection(self):
        try:
            connection = mysql.connector.connect(
                host=os.getenv('DB_HOST'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                database=os.getenv('DB_NAME')
            )
            
            # Create attendance table if it doesn't exist
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    registration_number VARCHAR(20) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    recognition_method VARCHAR(10) NOT NULL
                )
            """)
            connection.commit()
            
            return connection
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None
    
    def start_recognition(self, mode):
        self.current_mode = mode
        if not self.is_camera_on:
            # self.camera = cv2.VideoCapture("http://192.168.1.104:8080/video")
            self.camera = cv2.VideoCapture(0)
            self.is_camera_on = True
            self.update_camera()
    
    def update_camera(self):
        if self.is_camera_on:
            ret, frame = self.camera.read()
            if ret:
                self.frame_count += 1
                
                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 480))
                
                # Only process every nth frame
                if self.frame_count % self.process_every_n_frames == 0:
                    if self.current_mode == "face":
                        self.process_face_recognition(frame)
                    elif self.current_mode == "card":
                        self.process_card_recognition(frame)
                    self.last_processed_frame = frame.copy()
                else:
                    # Use the last processed frame for display
                    if self.last_processed_frame is not None:
                        frame = self.last_processed_frame
                
                # Convert frame to PhotoImage
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (800, 600))
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo
            
            # Schedule next update with a shorter delay
            self.root.after(10, self.update_camera)
    
    def process_face_recognition(self, frame):
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare faces with a lower tolerance for stricter matching
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.4)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            name = "Unknown"
            confidence = 0
            
            if True in matches:
                # Get the best match
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    
                    # Only mark attendance if confidence is high enough
                    if confidence > 0.6:
                        self.mark_attendance(name, "face")
                    
                        
            
            # Draw rectangle and name with confidence
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    
    def process_card_recognition(self, frame):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Extract text using Tesseract with specific configuration
            text = pytesseract.image_to_string(
                thresh,
                config='--psm 6 --oem 3 -c tessedit_char_whitelist=FS0123456789BSCS'
            )
            
            # Look for registration number pattern
            import re
            reg_pattern = r'[FS]\d{2}BSCS\d{3}'
            matches = re.findall(reg_pattern, text)
            
            if matches:
                reg_number = matches[0].upper()
                # Only process if it's a valid registration number
                if reg_number in self.valid_registration_numbers:
                    self.mark_attendance(reg_number, "card")
                    winsound.Beep(1000, 500)
                    # Draw green rectangle and text for valid card
                    cv2.rectangle(frame, (10, 10), (300, 50), (0, 255, 0), 2)
                    cv2.putText(frame, f"Valid Card: {reg_number}", (20, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                      # 1000Hz for 500ms
                    # Draw red rectangle and text for invalid card
                    cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 255), 2)
                    cv2.putText(frame, "Invalid Card", (20, 40),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Draw "No Card Detected" message
                cv2.putText(frame, "No Card Detected", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        except Exception as e:
            print(f"Error in card recognition: {e}")
            cv2.putText(frame, "Error in Card Recognition", (20, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    def load_attendance_records(self):
        if not self.db_connection:
            print("No database connection available")
            return
            
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT registration_number, DATE(timestamp) as date,
                       MIN(TIME(timestamp)) as time_in,
                       MAX(TIME(timestamp)) as time_out
                FROM attendance
                GROUP BY registration_number, DATE(timestamp)
                ORDER BY date DESC, time_in DESC
            """)
            
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Add records to tree
            for record in cursor.fetchall():
                self.tree.insert("", "end", values=record)
                
        except mysql.connector.Error as err:
            print(f"Error loading attendance records: {err}")

    def mark_attendance(self, reg_number, method):
        try:
            cursor = self.db_connection.cursor()
            current_time = datetime.now()
            # Check if attendance already marked today
            cursor.execute("""
                SELECT * FROM attendance 
                WHERE registration_number = %s 
                AND DATE(timestamp) = CURDATE()
            """, (reg_number,))
            
            existing_records = cursor.fetchall()
            
            if not existing_records:
                # First detection of the day - mark as time in
                cursor.execute("""
                    INSERT INTO attendance (registration_number, timestamp, recognition_method)
                    VALUES (%s, %s)
                """, (reg_number, current_time, method))
                self.status_label.config(text=f"Time In marked for {reg_number}")
            else:
                # Check if last detection was more than 5 seconds ago
                last_detection = existing_records[-1][2]  # timestamp column
                time_diff = (current_time - last_detection).total_seconds()
                
                if time_diff >= 5:  # Only record if 5 seconds have passed
                    cursor.execute("""
                        INSERT INTO attendance (registration_number, timestamp, recognition_method)
                        VALUES (%s, %s, %s)
                    """, (reg_number, current_time, method))
                    self.status_label.config(text=f"Time Out marked for {reg_number}")
                else:
                    self.status_label.config(text=f"Detection too soon for {reg_number}")
            
            self.db_connection.commit()
            
            # Update the attendance table
            self.load_attendance_records()
                
        except mysql.connector.Error as err:
            print(f"Database error: {err}")
            self.status_label.config(text="Error marking attendance")
    
    def _del_(self):
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
        if hasattr(self, 'db_connection') and self.db_connection is not None:
            self.db_connection.close()

    def load_known_faces(self):
        # Try to load from saved encodings first
        if os.path.exists('face_encodings.pkl'):
            try:
                with open('face_encodings.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print("Loaded face encodings from file")
                return
            except Exception as e:
                print(f"Error loading saved encodings: {e}")

        # If no saved encodings or loading failed, process images
        for student_dir in os.listdir("face_dataset"):
            student_path = os.path.join("face_dataset", student_dir)
            if os.path.isdir(student_path):
                for image_file in os.listdir(student_path):
                    if image_file.endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            image_path = os.path.join(student_path, image_file)
                            print(f"Processing image: {image_path}")

                            # Load image using PIL
                            face_image = Image.open(image_path)
                            
                            # Convert to RGB if needed
                            if face_image.mode != 'RGB':
                                face_image = face_image.convert('RGB')
                            
                            # Apply augmentations
                            augmented_images = augment_image(face_image)
                            
                            for aug_image in augmented_images:
                                # Convert PIL image to numpy array
                                face_array = np.array(aug_image)
                                
                                # Resize if needed
                                if face_array.shape[0] > 1000 or face_array.shape[1] > 1000:
                                    scale = min(1000 / face_array.shape[0], 1000 / face_array.shape[1])
                                    new_size = (int(face_array.shape[1] * scale), int(face_array.shape[0] * scale))
                                    face_array = cv2.resize(face_array, new_size)

                                # Detect face and encode
                                face_locations = face_recognition.face_locations(face_array)
                                if not face_locations:
                                    continue

                                face_encodings = face_recognition.face_encodings(face_array, face_locations)

                                if face_encodings:
                                    # Store multiple encodings for each person
                                    for encoding in face_encodings:
                                        self.known_face_encodings.append(encoding)
                                        self.known_face_names.append(student_dir)

                        except Exception as e:
                            print(f"Error processing {image_path}: {str(e)}")
                            continue

        # Save the encodings for future use
        try:
            with open('face_encodings.pkl', 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            print("Saved face encodings to file")
        except Exception as e:
            print(f"Error saving encodings: {e}")

    def retrain_faces(self):
        """Method to force retraining of faces"""
        if os.path.exists('face_encodings.pkl'):
            os.remove('face_encodings.pkl')
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_valid_registration_numbers(self):
        """Load valid registration numbers from student_cards directory"""
        try:
            for filename in os.listdir("car_dataset"):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    # Extract registration number from filename
                    reg_number = os.path.splitext(filename)[0]
                    self.valid_registration_numbers.add(reg_number)
            print(f"Loaded {len(self.valid_registration_numbers)} valid registration numbers")
        except Exception as e:
            print(f"Error loading valid registration numbers: {e}")

if __name__ == "__main__":
    root = ThemedTk(theme="arc")  # Using a modern theme
    app = AttendanceSystem(root)
    root.mainloop()
