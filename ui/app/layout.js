import "./globals.css";

export const metadata = {
  title: "Traffic Sign Recognition",
  description: "YOLOv8 + GTSRB CNN — live dashcam video detection — Mini-projet 6",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
