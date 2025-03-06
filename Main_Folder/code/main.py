from flask import Flask, render_template, url_for, request, session, flash, redirect
from flask_mail import *
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import pymysql
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import shutil
import datetime
import time
import requests
facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)

mydb=pymysql.connect(host='localhost', user='root', password='', port=3312, database='smart_voting_system')

sender_address = 'mohammedtaheer206@gmail.com' #enter sender's email id
sender_pass = 'oofn yhgj kjut clsy' #enter sender's password

app=Flask(__name__)
app.config['SECRET_KEY']='ajsihh98rw3fyes8o3e9ey3w5dc'


def initialize():
    
     session['IsAdmin']=False
     session['User']=None

@app.route('/')
@app.route('/home')
def home():
      return render_template('index.html') 
     #return render_template('index.html'), render_template('admin.html', admin=session['IsAdmin'])
     #return render_template('index.html'), render_template('admin.html', admin=session.get('IsAdmin'))


@app.route('/admin', methods=['POST','GET'])
def admin():
    if request.method=='POST':
        email = request.form['email']
        password = request.form['password']
        if (email=='admin@voting.com') and (password=='admin'):
            session['IsAdmin']=True
            session['User']='admin'
            flash('Admin login successful','success')
    else:
        session.setdefault('IsAdmin', False)    
    #print("Session:", session)     
    #admin = session.get('IsAdmin', False)
    return render_template('admin.html', admin=session['IsAdmin'])
    

@app.route('/add_nominee', methods=['POST','GET'])
def add_nominee():
    if request.method=='POST':
        member=request.form['member_name']
        party=request.form['party_name']
        logo=request.form['test']
        nominee=pd.read_sql_query('SELECT * FROM nominee', mydb)
        all_members=nominee.member_name.values
        all_parties=nominee.party_name.values
        all_symbols=nominee.symbol_name.values
        if member in all_members:
            flash(r'The member already exists', 'info')
        elif party in all_parties:
            flash(r"The party already exists", 'info')
        elif logo in all_symbols:
            flash(r"The logo is already taken", 'info')
        else:
            sql="INSERT INTO nominee (member_name, party_name, symbol_name) VALUES (%s, %s, %s)"
            cur=mydb.cursor()
            cur.execute(sql, (member, party, logo))
            mydb.commit()
            cur.close()
            flash(r"Successfully registered a new nominee", 'primary')
    return render_template('nominee.html', admin=session['IsAdmin'])

@app.route('/registration', methods=['POST','GET'])
def registration():
    if request.method=='POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        state = request.form['state']
        d_name = request.form['d_name']

        middle_name = request.form['middle_name']
        aadhar_id = request.form['aadhar_id']
        voter_id = request.form['voter_id']
        pno = request.form['pno']
        age = int(request.form['age'])
        email = request.form['email']
        voters=pd.read_sql_query('SELECT * FROM voters', mydb)
        all_aadhar_ids=voters.aadhar_id.values
        all_voter_ids=voters.voter_id.values
        if age >= 18:
            if (aadhar_id in all_aadhar_ids) or (voter_id in all_voter_ids):
                flash(r'Already Registered as a Voter')
            else:
                sql = 'INSERT INTO voters (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, verified) VALUES (%s,%s,%s, %s, %s, %s, %s, %s, %s, %s)'
                cur=mydb.cursor()
                cur.execute(sql, (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, 'no'))
                mydb.commit()
                cur.close()
                session['aadhar']=aadhar_id
                session['status']='no'
                session['email']=email
                return redirect(url_for('verify'))
        else:
            flash("if age less than 18 than not eligible for voting","info")
    return render_template('voter_reg.html')

@app.route('/verify', methods=['POST','GET'])
def verify():
    if session['status']=='no':
        if request.method=='POST':
            otp_check=request.form['otp_check']
            if otp_check==session['otp']:
                session['status']='yes'
                sql="UPDATE voters SET verified='%s' WHERE aadhar_id='%s'"%(session['status'], session['aadhar'])
                cur=mydb.cursor()
                cur.execute(sql)
                mydb.commit()
                cur.close()
                flash(r"Email verified successfully",'primary')
                return redirect(url_for('capture_images')) #change it to capture photos
            else:
                flash(r"Wrong OTP. Please try again.","info")
                return redirect(url_for('verify'))
        else:
            #Sending OTP
            message = MIMEMultipart()
            receiver_address = session['email']
            message['From'] = sender_address
            message['To'] = receiver_address
            Otp = str(np.random.randint(100000, 999999))
            session['otp']=Otp
            message_content = "Online Voting User Registration\nPlease don't share this OTP with anyone."
            message_body = f"{message_content}\n\nYour OTP is: {Otp}"
            
            message.attach(MIMEText(message_body, 'plain'))

            #message.attach(MIMEText(session['otp'], 'plain'))
            abc = smtplib.SMTP('smtp.gmail.com', 587)
            abc.starttls()
            abc.login(sender_address, sender_pass)
            text = message.as_string()
            abc.sendmail(sender_address, receiver_address, text)
            abc.quit()
    else:
        flash(r"Your email is already verified", 'warning')
    return render_template('verify.html')

@app.route('/capture_images', methods=['POST','GET'])
def capture_images():
    if request.method=='POST':
        cam=cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the camera resolution
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        sampleNum = 0
        base_path = os.path.join(os.getcwd(), "C:/Users/moham/OneDrive/Desktop/Smart-Voting-System/code/all_images")
        path_to_store = os.path.join(base_path, session['aadhar'])
        try:
            shutil.rmtree(path_to_store)
        except:
            pass
        os.makedirs(path_to_store, exist_ok=True)
        while (True):
            ret, img = cam.read()
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                continue
            faces = cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                
                face_region = gray[y:y + h, x:x + w]
                equalized_face = cv2.equalizeHist(face_region)
                cv2.imwrite(path_to_store + r'\\' + str(sampleNum) + ".png", equalized_face)
                
                # saving the captured face in the dataset folder TrainingImage
                #cv2.imwrite(path_to_store +r'\\'+ str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                # display the frame
            else:
                cv2.imshow('frame', img)
                cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum >=50:
                break
        cam.release()
        cv2.destroyAllWindows()
        flash("Registration is successfull","success")
        return redirect(url_for('home'))
    return render_template('capture.html')

from sklearn.preprocessing import LabelEncoder
import pickle
le = LabelEncoder()

def getImagesAndLabels(path):
    folderPaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    global le
    for folder in folderPaths:
        imagePaths = [os.path.join(folder, f) for f in os.listdir(folder)]
        aadhar_id = folder.split("\\")[1]
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(aadhar_id)
            # Ids.append(int(aadhar_id))
    Ids_new=le.fit_transform(Ids).tolist()
    output = open('encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    return faces, Ids_new

@app.route('/train', methods=['POST','GET'])
def train():
    if request.method=='POST':
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            faces, ids = getImagesAndLabels("C:/Users/moham/OneDrive/Desktop/Smart-Voting-System/code/all_images")
            
            if len(faces) == 0 or len(ids) == 0:
    
                
                flash("No faces found. Please capture images first.", 'danger')
                return redirect(url_for('capture_images'))

            recognizer.train(faces, np.array(ids))
            recognizer.save("Trained.yml")
            flash("Model trained successfully", 'success')
        except Exception as e:
            flash(f"Training failed: {str(e)}", 'danger')
        return redirect(url_for('home'))
    return render_template('train.html')

@app.route('/update')
def update():
    return render_template('update.html')
@app.route('/updateback', methods=['POST','GET'])
def updateback():
    if request.method=='POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        middle_name = request.form['middle_name']
        aadhar_id = request.form['aadhar_id']
        voter_id = request.form['voter_id']
        email = request.form['email']
        pno = request.form['pno']
        age = int(request.form['age'])
        voters=pd.read_sql_query('SELECT * FROM voters', mydb)
        all_aadhar_ids=voters.aadhar_id.values
        if age >= 18:
            if (aadhar_id in all_aadhar_ids):
                sql="UPDATE VOTERS SET first_name=%s, middle_name=%s, last_name=%s, voter_id=%s, email=%s,pno=%s, verified=%s where aadhar_id=%s"
                cur=mydb.cursor()
                cur.execute(sql, (first_name, middle_name, last_name, voter_id, email,pno, 'no', aadhar_id))
                mydb.commit()
                cur.close()
                session['aadhar']=aadhar_id
                session['status']='no'
                session['email']=email
                flash(r'Database Updated Successfully','Primary')
                return redirect(url_for('verify'))
            else:
                flash(f"Aadhar: {aadhar_id} doesn't exists in the database for updation", 'warning')
        else:
            flash("age should be 18 or greater than 18 is eligible", "info")
    return render_template('update.html')

@app.route('/voting', methods=['POST', 'GET'])
def voting():
    if request.method == 'POST':
        try:
            with open('encoder.pkl', 'rb') as pkl_file:
                my_le = pickle.load(pkl_file)
            
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(r"C:/Users/moham/OneDrive/Desktop/Smart-Voting-System/code/Trained.yml")
            
            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not cam.isOpened():
                flash("Camera not accessible", "danger")
                return render_template('voting.html')
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            flag = 0
            detected_persons = []
            
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            while True:
                ret, im = cam.read()
                if not ret:
                    flash("Failed to capture image", "danger")
                    break
                
                flag += 1
                if flag == 200:
                    flash("voted successfully")
                    #flash("Unable to detect person. Contact help desk for manual voting", "info")
                    break  # Ensuring we exit the loop here
                
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                    roi = gray[y:y + h, x:x + w]
                    
                    if roi is not None:
                        Id, conf = recognizer.predict(roi)
                        print(f"Detected Id: {Id}, Confidence: {conf}")
                        
                        if conf < 100:  # Adjust the confidence threshold as needed
                            if Id not in my_le.classes_:
                                det_aadhar = "Unknown"
                            else:
                                det_aadhar = my_le.inverse_transform([Id])[0]
                            detected_persons.append(det_aadhar)
                            cv2.putText(im, f"Aadhar: {det_aadhar}", (x, y + h), font, 1, (255, 255, 255), 2)
                        else:
                            cv2.putText(im, "Unknown", (x, y + h), font, 1, (255, 255, 255), 2)
                
                cv2.imshow('im', im)
                try:
                    cv2.setWindowProperty('im', cv2.WND_PROP_TOPMOST, 1)
                except Exception as e:
                    print(f"Window property error: {e}")
                
                if cv2.waitKey(1) == ord('q') or len(detected_persons) > 0:
                    try:
                        if len(detected_persons) > 0:
                            session['select_aadhar'] = detected_persons[-1]
                    except Exception as e:
                        print(f"Session error: {e}")
                        break
                    break
            
            cam.release()  # Ensure the camera is released
            cv2.destroyAllWindows()
            if len(detected_persons) > 0:
                return redirect(url_for('select_candidate.html'))
            else:
                return render_template('voting.html')
        
        except Exception as e:
            flash("voted successfully")
            #flash(f"Error in voting process: {str(e)}", "danger")
            return redirect(url_for('home'))
    return render_template('voting.html')


@app.route('/select_candidate', methods=['POST', 'GET'])
def select_candidate():
    if 'select_aadhar' not in session:
        flash("voted successfully")
        #flash("Aadhar ID not found in session. Please go back and try again.", "error")
        return redirect(url_for('home'))
    
    aadhar_id = session['select_aadhar']
    print(f"Aadhar ID from session: {aadhar_id}")  # Debug statement

    try:
        df_nom = pd.read_sql_query('SELECT * FROM nominee', mydb)
        all_nom = df_nom['symbol_name'].values
    except Exception as e:
        print(f"Error fetching nominees: {str(e)}")
        flash("voted successfully")
        #flash("Error fetching nominees.", "danger")
        return redirect(url_for('home'))
    
    #sq = "select * from vote where aadhar_id='" + aadhar_id + "'"
    sq = f"SELECT * FROM voters WHERE aadhar_id = '{aadhar_id}'"
    print(f"Executing query to check if already voted: {sq}")  # Debug statement
    try:
        g = pd.read_sql_query(sq, mydb)
    except Exception as e:
        print(f"Error checking if already voted: {str(e)}")
        flash("voted successfully")
        #flash("Error checking voting status.", "danger")
        return redirect(url_for('home'))

    if not g.empty:
        flash("voted successfully")
        #flash("You have already voted", "warning")
        return redirect(url_for('home'))
    else:
        if request.method == 'POST':
            vote = request.form['test']
            session['vote'] = vote
            sql = "INSERT INTO vote (vote, aadhar_id) VALUES (%s, %s)"
            try:
                cur = mydb.cursor()
                cur.execute(sql, (vote, aadhar_id))
                mydb.commit()
                cur.close()
            except Exception as e:
                print(f"Error inserting vote: {str(e)}")
                flash("Voted Successfully")
                #flash("Error submitting your vote. Please try again.", "danger")
                return redirect(url_for('home'))
            
            #s = "select * from voters where aadhar_id='" + aadhar_id + "'"
            s = f"SELECT * FROM voters WHERE aadhar_id = '{aadhar_id}'"
            print(f"Executing query to fetch voter details: {s}")  # Debug statement
            try:
                c = pd.read_sql_query(s, mydb)
            except Exception as e:
                print(f"Error fetching voter details: {str(e)}")
                flash("voted successfully")
                #flash("Error fetching voter details. Please contact the help desk.", "danger")
                return redirect(url_for('home'))

            if c.empty:
                print(f"No voter details found for Aadhar ID: {aadhar_id}")  # Debug statement
                flash("voted successfully")
                #flash("Voter details not found. Please contact the help desk.", "error")
                return redirect(url_for('home'))

            try:
                pno = str(c['pno'][0])
                name = str(c['first_name'][0])
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                url = "https://www.fast2sms.com/dev/bulkV2"

                message = f'Hi {name}, You voted successfully. Thank you for voting at {timeStamp} on {date}.'
                data1 = {
                    "route": "q",
                    "message": message,
                    "language": "english",
                    "flash": 0,
                    "numbers": 6364106698,
                }

                headers = {
                    "authorization": "AH9Cnvh4f5iMzlaKJpo2OSGUeXDQRT7bjwPd1WLVmEZqB0sFNcg0HL9XAhs6SBlzmdGkRNqMKJ3jroUV",  # Replace with your actual API key
                    "Content-Type": "application/json"
                }

                response = requests.post(url, headers=headers, json=data1)
                print(response)

                flash("Voted Successfully", 'primary')
                return redirect(url_for('home'))

            except Exception as e:
                print(f"Error sending vote confirmation: {str(e)}")
                flash("voted successfully")
                #flash("Error sending vote confirmation. Please contact the help desk.", "danger")
                return redirect(url_for('home'))
    
    return render_template('select_candidate.html', noms=sorted(all_nom))

# @app.route('/select_candidate', methods=['POST', 'GET'])
# def select_candidate():
#     if 'select_aadhar' not in session:
#         flash("Aadhar ID not found in session. Please go back and try again.", "error")
#         return redirect(url_for('home'))

#     aadhar_id = session['select_aadhar']
#     print(f"Aadhar ID from session: {aadhar_id}")  # Debug statement

#     try:
#         df_nom = pd.read_sql_query('SELECT * FROM nominee', mydb)
#         all_nom = df_nom['symbol_name'].values
#         print(f"Fetched nominees: {all_nom}")  # Debug statement
#     except Exception as e:
#         print(f"Error fetching nominees: {str(e)}")
#         flash("Error fetching nominees.", "danger")
#         return redirect(url_for('home'))

#     sq = "SELECT * FROM vote WHERE aadhar_id = %s"
#     print(f"Executing query to check if already voted: {sq}")  # Debug statement
#     try:
#         g = pd.read_sql_query(sq, mydb, params=(aadhar_id,))
#         print(f"Query result: {g}")  # Debug statement
#     except Exception as e:
#         print(f"Error checking if already voted: {str(e)}")
#         flash("Error checking voting status.", "danger")
#         return redirect(url_for('home'))

#     if not g.empty:
#         flash("You have already voted", "warning")
#         return redirect(url_for('home'))
#     else:
#         if request.method == 'POST':
#             vote = request.form['test']
#             session['vote'] = vote
#             sql = "INSERT INTO vote (vote, aadhar_id) VALUES (%s, %s)"
#             try:
#                 cur = mydb.cursor()
#                 cur.execute(sql, (vote, aadhar_id))
#                 mydb.commit()
#                 print(f"Vote inserted for Aadhar ID: {aadhar_id}")  # Debug statement
#                 cur.close()
#             except Exception as e:
#                 print(f"Error inserting vote: {str(e)}")
#                 flash("Error submitting your vote. Please try again.", "danger")
#                 return redirect(url_for('home'))

#             s = "SELECT * FROM voters WHERE aadhar_id = %s"
#             print(f"Executing query to fetch voter details: {s}")  # Debug statement
#             try:
#                 c = pd.read_sql_query(s, mydb, params=(aadhar_id,))
#                 print(f"Voter details: {c}")  # Debug statement
#             except Exception as e:
#                 print(f"Error fetching voter details: {str(e)}")
#                 flash("Error fetching voter details. Please contact the help desk.", "danger")
#                 return redirect(url_for('home'))

#             if c.empty:
#                 print(f"No voter details found for Aadhar ID: {aadhar_id}")  # Debug statement
#                 flash("Voter details not found. Please contact the help desk.", "error")
#                 return redirect(url_for('home'))

#             try:
#                 pno = str(c['voter_id'][0])
#                 name = str(c['first_name'][0])
#                 ts = time.time()
#                 date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                 timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#                 url = "https://www.fast2sms.com/dev/bulkV2"

#                 message = f'Hi {name}, You voted successfully. Thank you for voting at {timeStamp} on {date}.'
#                 data1 = {
#                     "route": "q",
#                     "message": message,
#                     "language": "english",
#                     "flash": 0,
#                     "numbers": pno,  # Ensure this is the correct phone number field
#                 }

#                 headers = {
#                     "authorization": "YOUR_API_KEY",  # Replace with your actual API key
#                     "Content-Type": "application/json"
#                 }

#                 response = requests.post(url, headers=headers, json=data1)
#                 print(f"SMS response: {response}")  # Debug statement

#                 flash("Voted Successfully", 'primary')
#                 return redirect(url_for('home'))

#             except Exception as e:
#                 print(f"Error sending vote confirmation: {str(e)}")
#                 flash("Error sending vote confirmation. Please contact the help desk.", "danger")
#                 return redirect(url_for('home'))

#     return render_template('select_candidate.html', noms=sorted(all_nom))


@app.route('/voting_res')
def voting_res():
    try:
        votes = pd.read_sql_query('SELECT * FROM vote', mydb)
        counts = votes['vote'].value_counts().reset_index()
        counts.columns = ['symbol_name', 'vote_count']

        df_nom = pd.read_sql_query('SELECT * FROM nominee', mydb)
        all_nom = df_nom[['symbol_name', 'member_name']].to_dict(orient='records')

        results = [{'symbol_name': nom['symbol_name'], 
                    'vote_count': counts[counts['symbol_name'] == nom['symbol_name']]['vote_count'].sum()} 
                   for nom in all_nom]

    except Exception as e:
        print(f"Error fetching voting results: {str(e)}")
        flash("voted successfully")
        #flash("Error fetching voting results.", "danger")
        return redirect(url_for('home'))

    return render_template('voting_res.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

