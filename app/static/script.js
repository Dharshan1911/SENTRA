const btnImage = document.getElementById('btn-image');
const btnVideo = document.getElementById('btn-video');
const btnWebcam = document.getElementById('btn-webcam');
const fileInput = document.getElementById('file-input');
const uploadPrompt = document.getElementById('upload-prompt');
const resultImg = document.getElementById('result-img');
const inputVideo = document.getElementById('input-video');
const canvas = document.getElementById('canvas');

const statPeople = document.getElementById('stat-people');
const statCompliant = document.getElementById('stat-compliant');
const statViolations = document.getElementById('stat-violations');
const statFps = document.getElementById('stat-fps');
const ppeStatusList = document.getElementById('ppe-status-list');
const safetyStatus = document.getElementById('safety-status');

let currentMode = 'image';
let ws = null;
let videoInterval = null;

// Positive and Negative classes
const ppeTypes = ["Helmet", "Gloves", "Vest", "Boots", "Goggles"];

btnImage.addEventListener('click', () => setMode('image'));
btnVideo.addEventListener('click', () => setMode('video'));
btnWebcam.addEventListener('click', () => setMode('webcam'));

function setMode(mode) {
    currentMode = mode;
    
    btnImage.classList.toggle('active', mode === 'image');
    btnVideo.classList.toggle('active', mode === 'video');
    btnWebcam.classList.toggle('active', mode === 'webcam');
    
    stopWebcam();
    stopVideo();
    
    resultImg.style.display = 'none';
    inputVideo.style.display = 'none';
    
    if (mode === 'image') {
        uploadPrompt.style.display = 'flex';
        uploadPrompt.querySelector('p').innerText = "Drag and drop image here or click to upload";
        fileInput.accept = "image/*";
    } else if (mode === 'video') {
        uploadPrompt.style.display = 'flex';
        uploadPrompt.querySelector('p').innerText = "Drag and drop video here or click to upload";
        fileInput.accept = "video/*";
    } else if (mode === 'webcam') {
        uploadPrompt.style.display = 'none';
        startWebcam();
    }
}

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        if (currentMode === 'image') {
            processImage(file);
        } else if (currentMode === 'video') {
            processVideo(file);
        }
    }
});

function updateStats(data) {
    statPeople.innerText = data.persons;
    statCompliant.innerText = Math.max(0, data.persons - data.violations);
    statViolations.innerText = data.violations;
    statFps.innerText = data.fps;

    // Update PPE status list
    ppeStatusList.innerHTML = '';
    
    const detectedClasses = data.detections.map(d => d.class);
    
    // Simple logic based on detected classes
    ppeTypes.forEach(ppe => {
        const div = document.createElement('div');
        div.className = 'ppe-item';
        
        const nameSpan = document.createElement('span');
        nameSpan.innerText = ppe;
        
        const statusSpan = document.createElement('span');
        
        // Find highest confidence for this PPE
        const dets = data.detections.filter(d => d.class === ppe);
        if (dets.length > 0) {
            const maxConf = Math.max(...dets.map(d => d.confidence));
            statusSpan.innerText = `✓ DETECTED ${Math.round(maxConf * 100)}%`;
            statusSpan.className = 'ppe-detected';
        } else {
            statusSpan.innerText = `--`;
            statusSpan.className = 'ppe-missing';
        }
        
        div.appendChild(nameSpan);
        div.appendChild(statusSpan);
        ppeStatusList.appendChild(div);
    });

    if (data.is_safe) {
        safetyStatus.className = 'compliant';
        safetyStatus.innerText = '✓ COMPLIANT';
    } else {
        safetyStatus.className = 'non-compliant';
        safetyStatus.innerText = '⚠ NON-COMPLIANT';
    }
}

async function processImage(file) {
    uploadPrompt.style.display = 'none';
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/predict/image', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            return;
        }

        resultImg.src = data.image;
        resultImg.style.display = 'block';
        updateStats(data);
    } catch (err) {
        console.error(err);
    }
}

function processVideo(file) {
    uploadPrompt.style.display = 'none';
    const url = URL.createObjectURL(file);
    inputVideo.src = url;
    inputVideo.style.display = 'none'; // hide the raw video, we will show the processed frames
    resultImg.style.display = 'block'; // show the processed frames here

    inputVideo.onloadedmetadata = () => {
        canvas.width = inputVideo.videoWidth;
        canvas.height = inputVideo.videoHeight;
        inputVideo.play();
        
        // Process frames
        videoInterval = setInterval(async () => {
            if (inputVideo.paused || inputVideo.ended) {
                clearInterval(videoInterval);
                return;
            }
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(inputVideo, 0, 0, canvas.width, canvas.height);
            
            // Convert to blob
            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append('file', blob, 'frame.jpg');
                
                try {
                    const response = await fetch('/api/predict/image', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (!data.error) {
                        resultImg.src = data.image;
                        updateStats(data);
                    }
                } catch(e) {
                    console.error(e);
                }
            }, 'image/jpeg', 0.8);
            
        }, 1000 / 5); // Target ~5 FPS for video to not overload backend
    };
}

function stopVideo() {
    if (videoInterval) {
        clearInterval(videoInterval);
        videoInterval = null;
    }
    inputVideo.pause();
    inputVideo.src = "";
}

function startWebcam() {
    resultImg.style.display = 'block';
    
    // Connect to WebSocket
    ws = new WebSocket(`ws://${window.location.host}/ws/webcam`);
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.error) {
            alert(data.error);
            stopWebcam();
            return;
        }
        resultImg.src = data.image;
        updateStats(data);
    };
    
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        inputVideo.srcObject = stream;
        inputVideo.play();
        
        inputVideo.onloadedmetadata = () => {
            canvas.width = inputVideo.videoWidth;
            canvas.height = inputVideo.videoHeight;
            
            videoInterval = setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(inputVideo, 0, 0, canvas.width, canvas.height);
                    const dataURL = canvas.toDataURL('image/jpeg', 0.7);
                    ws.send(dataURL);
                }
            }, 1000 / 10); // Target 10 FPS for webcam
        };
    }).catch(err => {
        alert("Webcam access denied or unavailable.");
        console.error(err);
    });
}

function stopWebcam() {
    if (videoInterval) {
        clearInterval(videoInterval);
        videoInterval = null;
    }
    if (ws) {
        ws.close();
        ws = null;
    }
    if (inputVideo.srcObject) {
        inputVideo.srcObject.getTracks().forEach(track => track.stop());
        inputVideo.srcObject = null;
    }
}
