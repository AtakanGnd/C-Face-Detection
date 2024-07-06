using Emgu.CV;
using System;
using System.Windows.Forms;
using System.Drawing;
using System.IO;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Reg;

namespace FaceDetection
{
    public partial class Form1 : Form
    {

        private VideoCapture _capture;
        private CascadeClassifier _faceCascade;
        private Timer _timer;

        public Form1()
        {
            InitializeComponent();
            string basePath = System.IO.Path.GetFullPath(@"..\..\..\");
            _faceCascade = new CascadeClassifier(Path.Combine(basePath, "haarcascade_frontalface_default.xml"));
            _capture = new VideoCapture(0); // 0, varsayılan kamerayı kullanır
            _capture.ImageGrabbed += ProcessFrame;

            _timer = new Timer
            {
                Interval = 30
            };
            _timer.Tick += (sender, args) => _capture.Start();
            _timer.Start();
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            var frame = new Mat();
            _capture.Retrieve(frame);

            var grayFrame = new Mat();
            CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);
            CvInvoke.EqualizeHist(grayFrame, grayFrame);

            var faces = _faceCascade.DetectMultiScale(grayFrame, 1.1, 10);

            foreach (var face in faces)
            {
                // Yüzün tüm çevresi
                DrawEllipsePoints(frame, face, 50);

                // Gözler
                var leftEye = new Rectangle(face.X + face.Width / 4 - 10, face.Y + face.Height / 4 + 5, face.Width / 5, face.Height / 5);
                var rightEye = new Rectangle(face.X + 3 * face.Width / 4 - face.Width / 5, face.Y + face.Height / 4, face.Width / 5, face.Height / 5);
                DrawEllipsePoints(frame, leftEye, 20);
                DrawEllipsePoints(frame, rightEye, 20);

                // Burun
                var nose = new Rectangle(face.X + face.Width / 3, face.Y + face.Height / 2, face.Width / 3, face.Height / 5);
                DrawEllipsePoints(frame, nose, 20);

                // Ağız
                var mouth = new Rectangle(face.X + face.Width / 4, face.Y + 3 * face.Height / 4 - 5, face.Width / 2, face.Height / 5);
                DrawEllipsePoints(frame, mouth, 30);
            }

            pictureBox1.Image = frame.ToBitmap();
        }

        private void DrawEllipsePoints(Mat frame, Rectangle rect, int pointsCount)
        {
            for (int i = 0; i < pointsCount; i++)
            {
                var angle = 2 * Math.PI * i / pointsCount;
                var x = rect.X + rect.Width / 2 + (int)(rect.Width / 2 * Math.Cos(angle));
                var y = rect.Y + rect.Height / 2 + (int)(rect.Height / 2 * Math.Sin(angle));
                CvInvoke.Circle(frame, new Point(x, y), 3, new MCvScalar(0, 0, 255), -1);
            }
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            _capture.Dispose();
            base.OnFormClosed(e);
        }

        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            _capture.Dispose();
        }
    }
}
