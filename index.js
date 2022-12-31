const cv = require('@u4/opencv4nodejs');
const path = require("path");
const jimp = require("jimp");
const fs = require("fs");
const axios = require("axios");
const url = require("url");
//express
const express = require("express");
var app = express();
const { Webhook } = require('discord-webhook-node');
const hook = new Webhook("https://discord.com/api/webhooks/1058711410458238996/r-1XUYiglojxLNcQUUHs-mZiyEHyLckY9eF_cWz6fH40MDEFDyVYglOSYKm-IVcDzeuW");
//RTTEX
class RTFileHeader {
    constructor() {
        this.fileTypeId;
        this.version;
        this.reversed;
    }

    serialize(buffer, pos = 0) {
        this.fileTypeId = buffer.subarray(pos, 6).toString();
        pos += 6;

        this.version = buffer.readInt8(pos);
        pos += 1;

        // reverse : uint8_t
        pos += 1;
        return pos;
    }
}

class RTPackHeader {
    constructor() {
        this.rtFileHeader;
        this.compressedSize;
        this.decompressedSize;
        this.compressionType;
        this.reversed;
    }

    serialize(buffer, pos = 0) {
        this.rtFileHeader = new RTFileHeader();
        pos = this.rtFileHeader.serialize(buffer, pos);

        this.compressedSize = buffer.readInt32LE(pos);
        pos += 4;

        this.decompressedSize = buffer.readInt32LE(pos);
        pos += 4;

        this.compressionType = buffer.readInt8(pos);
        pos += 1;

        // reverse : uint8_t[15]
        pos += 15;
        return pos;
    }
}

class RTTEXHeader {
    constructor() {
        this.rtFileHeader;
        this.height;
        this.width;
        this.format;
        this.originalHeight;
        this.originalWidth;
        this.usesAlpha;
        this.aleardyCompressed;
        this.reversedFlags;
        this.mipmapCount;
        this.reversed;
    }

    serialize(buffer, pos = 0) {
        this.rtFileHeader = new RTFileHeader();
        pos = this.rtFileHeader.serialize(buffer, pos);

        this.height = buffer.readInt32LE(pos);
        pos += 4;

        this.width = buffer.readInt32LE(pos);
        pos += 4;

        this.format = buffer.readInt32LE(pos);
        pos += 4;

        this.originalHeight = buffer.readInt32LE(pos);
        pos += 4;

        this.originalWidth = buffer.readInt32LE(pos);
        pos += 4;

        this.usesAlpha = buffer.readInt8(pos);
        pos += 1;

        this.aleardyCompressed = buffer.readInt8(pos);
        pos += 1;

        // reservedFlags : unsigned char
        pos += 2;

        this.mipmapCount = buffer.readInt32LE(pos);
        pos += 4;

        // reserved : int[16]
        pos += 64;
        return pos;
    }
}

class RTTEXMipHeader {
    constructor() {
        this.height;
        this.width;
        this.dataSize;
        this.mipLevel;
        this.reversed;
    }

    serialize(buffer, pos = 0) {
        this.height = buffer.readInt32LE(pos);
        pos += 4;

        this.width = buffer.readInt32LE(pos);
        pos += 4;

        this.dataSize = buffer.readInt32LE(pos);
        pos += 4;

        this.mipLevel = buffer.readInt32LE(pos);
        pos += 4;

        // reversed : int[2]
        pos += 8;
        return pos;
    }
}

class RTTEX {
    constructor(buffer, pos = 0) {
        this.rttexHeader = new RTTEXHeader();
        this.rtpackHeader = new RTPackHeader();
        this.buffer = buffer;

        if (this.buffer.subarray(pos, 6).toString() == "RTPACK") {
            let temp_pos = this.rtpackHeader.serialize(this.buffer, pos);
            this.buffer = zlib.inflateSync(this.buffer.subarray(temp_pos, this.buffer.length));
        }

        this.pos = this.rttexHeader.serialize(this.buffer, pos);
    }

    async rawData(flipVertical = true) {
        return new Promise(resolve => {
            if (this.rttexHeader.format != 5121) {
                resolve(null);
            }

            let posBefore = this.pos;
            for (let i = 0; i < this.rttexHeader.mipmapCount; i++) {
                let mipHeader = new RTTEXMipHeader();
                this.pos = mipHeader.serialize(this.buffer, this.pos);
                let mipData = this.buffer.subarray(this.pos, this.pos + mipHeader.dataSize);

                this.pos = posBefore;
                
                if (flipVertical) {
                    new jimp(mipHeader.width, mipHeader.height, (err, image) => {
                        if (err) {
                            // throw err;
                            resolve(null);
                        }
                        
                        image.bitmap.data.set(mipData);
                        image.flip(false, true);
                        resolve(image.bitmap.data);
                    });
                }
            
                resolve(mipData);
            }

            resolve(null); 
        });
    }

    async write(path, flipVertical = true) {
        return new Promise(async (resolve) => {
            new jimp(this.rttexHeader.width, this.rttexHeader.height, async (err, image) => {
                if (err) {
                    // throw err;
                    resolve(false);
                }

                let ret = await this.rawData();
                image.bitmap.data.set(ret);
                image.flip(false, flipVertical);
                image.write(path, (err) => {
                    if (err) {
                        // throw err;
                        resolve(false);
                    }

                    resolve(true);
                });
            });
        });
    }
}
//Path Config
const darknetPath = "./newconfig";

const cfgFile = path.resolve(darknetPath, "custom-yolov4-tiny-detector.cfg");
const weightsFile = path.resolve(darknetPath, "custom-yolov4-tiny-detector_best.weights");
const labelsFile = path.resolve(darknetPath, "coco.names");

//threshold
const minConfidence = 0.88;
const nmsThreshold = 0.3;



const labels = fs
  .readFileSync(labelsFile)
  .toString()
  .split("\n");

  //read darknet model
const net = cv.readNetFromDarknet(cfgFile, weightsFile);
const allLayerNames = net.getLayerNames();
const unconnectedOutLayers = net.getUnconnectedOutLayers();

const layerNames = unconnectedOutLayers.map(layerIndex => {
    return allLayerNames[layerIndex - 1];
});

async function ProcessImage(bufferimage){
    const img = cv.imread(bufferimage);
    //object detection model works with 416 x 416 images
    const size = new cv.Size(640, 640);
    const vec3 = new cv.Vec(0, 0, 0);
    
    //const imgHeight = 640;
    //const imgWidth = 640;

    const [imgHeight, imgWidth] = img.sizes; //get img hight and width [ 256, 512 ]
    
    // network accepts blobs as input
    const inputBlob = cv.blobFromImage(img, 1 / 255.0, size, vec3, true,false);
    net.setInput(inputBlob);
  
    console.time("net.forward");
    // forward pass input through entire network
    const layerOutputs = net.forward(layerNames);
    console.timeEnd("net.forward");
  
    let boxes = [];
    let confidences = [];
    let classIDs = [];
  
    layerOutputs.forEach(mat => {
      const output = mat.getDataAsArray();
      output.forEach(detection => {
        const scores = detection.slice(5);
        const classId = scores.indexOf(Math.max(...scores));
        const confidence = scores[classId];
  
        if (confidence > minConfidence) {
          const box = detection.slice(0, 4);
  
          const centerX = parseInt(box[0] * imgWidth);
          const centerY = parseInt(box[1] * imgHeight);
          const width = parseInt(box[2] * imgWidth);
          const height = parseInt(box[3] * imgHeight);
  
          const x = parseInt(centerX - width / 2);
          const y = parseInt(centerY - height / 2);
  
          boxes.push(new cv.Rect(x, y, width, height));
          confidences.push(confidence);
          classIDs.push(classId);
  
          const indices = cv.NMSBoxes(
            boxes,
            confidences,
            minConfidence,
            nmsThreshold
          );
  
          indices.forEach(i => {
            const rect = boxes[i];
  
            const pt1 = new cv.Point(rect.x, rect.y);
            const pt2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
            const rectColor = new cv.Vec(255, 0, 255);
            const rectThickness = 1;
            const rectLineType = cv.LINE_8;
  
            // draw the rect for the object
            img.drawRectangle(pt1, pt2, rectColor, rectThickness, rectLineType);
  
            const text = labels[classIDs[i]];
            const org = new cv.Point(rect.x, rect.y + 15);
            const fontFace = cv.FONT_HERSHEY_SIMPLEX;
            const fontScale = 0.5;
            const textColor = new cv.Vec(123, 123, 255);
            const thickness = 2;

            // put text on the object
            const line1 = new cv.Point(rect.x + rect.width / 2, rect.y + rect.height);
            const line2 = new cv.Point(rect.x + rect.width / 2, 256);
            const lineColor = new cv.Vec(255, 255, 255);

            img.drawLine(line1,line2,lineColor,1,rectLineType);
            //img.putText(text, org, fontFace, fontScale, textColor, thickness);
            solvedx = rect.x + rect.width/2 + parseFloat(-1.5) + 1;
            
            widthimgpos = rect.width;
            heightimgpos = rect.height;

          if(rect.width >= 90 && rect.height >= 90){
                whatSize = "WOW BIGGG";
                calculat = parseFloat(solvedx)/573;
            }else if(rect.width >= 85 && rect.height >= 85){
                whatSize = "VERY BIG";
                calculat = parseFloat(solvedx)/568;
            }else if(rect.width >= 80 && rect.height >= 80){
                whatSize = "Little Big";
                calculat = parseFloat(solvedx)/585;
            }else if(rect.width >= 75 && rect.height >= 75){
                whatSize = "Big";
                calculat = parseFloat(solvedx)/569;
            }else if(rect.width >= 65 && rect.height >= 65){
                whatSize = "Almost Big";
                calculat = (parseFloat(solvedx))/570;
            }else if(rect.width >= 55 && rect.height >= 55){
                whatSize = "Large";
                calculat = (parseFloat(solvedx))/596;
            }else if(rect.width >= 45 && rect.height >= 45){
                whatSize = "Medium";
                calculat = (parseFloat(solvedx)+24)/575;
            }else{
                whatSize = "Small";
                calculat = (parseFloat(solvedx))/585;
            }

            if (rect.width >= 100 && rect.height >= 100 && solvedx >= 450) {
                whatSize = "FORCED RIGHT";
                calculat = 0.821875;
            }
          });
        }
      });
});
console.log(whatSize , widthimgpos , heightimgpos , solvedx);
console.log(calculat);
//cv.imshow("wpw",img);
//cv.waitKey();

return calculat = parseFloat(calculat).toFixed(6);
}

class PuzzleCaptchaSolver {
    constructor(rttex) {
        this.rttex = rttex.rttexHeader;
        this.rttexData = rttex;
        this.pixels = null;
        this.pixelsFiltered = new Uint8Array(this.rttex.height * this.rttex.width);
        this.filterDistance = 16;
        this.yellowLineCount = 0;
    }

    async loadAllPixel() {
        return new Promise(async (resolve, reject) => {
            this.pixels = await this.rttexData.rawData(true);
            resolve();
        });
    }

}

const axiosRequest = axios.create({
    responseType: "arraybuffer",
    baseURL: url,
    headers: {
        "Connection": "keep-alive",
        "Keep-Alive": "timeout=1500, max=100"
    }
});

app.use(express.json())
app.listen(3000, () => {
    console.log("Server running on port 3000");
});

app.get('/', async (req, res)=> {
    console.time("Start_Query");
    await axiosRequest.get(`https://ubistatic-a.akamaihd.net/0098/captcha/generated/${req.query.puzzeluid}-PuzzleWithMissingPiece.rttex`, {
        headers: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5097.0 Safari/537.36"
        }
    }).then(async response => {
        let buffer = Buffer.from(response.data, "binary");
        let rttex = new RTTEX(buffer);
        await rttex.write(`./${req.query.puzzeluid}.png`);
        let jawaban = await ProcessImage(`./${req.query.puzzeluid}.png`);
        
        hook.send(`captcha_answer|${jawaban}|CaptchaID|${req.query.puzzelid}`);
        res.send(`captcha_answer|${jawaban}|CaptchaID|${req.query.puzzelid}`);
    })
    .catch(ex => {
        console.log(`${ex}`);
        res.send(`captcha_answer|0,821875|CaptchaID|${req.query.puzzelid}`);
        hook.send(`captcha_answer|failed|CaptchaID|${req.query.puzzelid}`);
    });
    console.timeEnd("Start_Query");
    try {
         fs.unlinkSync(`./${req.query.puzzeluid}.png`)
       } catch(err) {
         console.error(err)
       }
});
//console.log(req.query.puzzelid);
//res.send(`captcha_answer|${123123123}|CaptchaID|${req.query.puzzelid}`);
//info

