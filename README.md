<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/xinrui98/ClimbAI">
    <img src="assets/climbAI-logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">ClimbAI</h3>

  <p align="center">
    An AI solution to enable visually impaired individuals to participate in rock climbing
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#ai-and-algorithm">AI and Algorithm</a>
    </li>
    <li>
      <a href="#user-applications">User Applications</a>
    </li>
    <li>
      <a href="#cloud-infrastructure">Cloud Infrastructure</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This project aims to create a technology-assistive solution that will enable visually impaired individuals to participate in rock climbing. The proposed solution will leverage computer vision technology to help visually impaired climbers navigate rock holds by analysing video streams and providing real-time guidance via audio means. The project will involve the development of custom software to process the video streams and generate the necessary guidance, as well as the creation of a cloud-based infrastructure to enable seamless deployment of the solution. Ultimately, the goal of this project is to promote inclusion and accessibility in the sport of rock climbing, allowing visually impaired individuals to enjoy the physical and mental benefits of this challenging activity.

Overall, the project deliverables is separated into 3 components: AI & Algorithms, User Applications, Cloud Infrastructure, which are taken charge by each team member: Boon Kong, Xinrui, Riyan respectively. The overall software architecture is as follows:

[![Product Name Screen Shot][product-screenshot]](https://example.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

This section outlines any major frameworks/libraries used to bootstrap our project.

- [![React][react.js]][react-url]
- [![Android][android]][android-url]
- [![watchOS][watchos]][watch-url]
- [![pyTortch][pytorch]][pytorch-url]
- [![detectron2][detectron2]][detectron2-url]
- [![mediapipe][mediapipe]][mediapipe-url]
- [![aws][aws]][aws-url]
- [![mq][mq]][mq-url]
- [![ant-media][ant-media]][ant-media-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- AI and Algorithm -->

# AI and Algorithm

BK

# User Applications

## Apple Watch

Serves to trigger the start of our cloud AI inference and track progress through timing and health statistics

### Technical Details

1. Our Apple Watch app is developed using SwiftUI and integrates Apple HealthKit to retrieve user health information.
2. Accessibility was a core consideration during the design process, ensuring that the app features clear, spoken instructions that can be easily followed by visually impaired users.
3. We use HTTP API calls to populate a cloud database with user data, including information on time, calories, and heart rate.

## Web Dashboard [https://climbai.cloud](https://climbai.cloud)

Mainly meant for the climber's coach to keep track of the climber's performance through weekly, monthly, yearly overviews

### Technical Details

1. Our Web Dashboard is built using React and JavaScript, providing a modern and responsive user interface.
2. We leverage the power of AWS Amplify to seamlessly integrate and access cloud databases, enabling us to easily store and retrieve data from a wide range of sources.

## Android App

Serves as a portable client application, allowing the climber to live stream the current climbing session, get real-time inference from the cloud AI on the next recommended rock hold, and receive audio feedback from the phone itself.

### Technical Details

1. Our Android app utilizes the Real-Time Messaging Protocol (RTMP) to transmit live footage of the climber's data to our cloud-based AI for inference, enabling real-time analysis and guidance.
2. We leverage multi-threading and asynchronous programming techniques to ensure that the app can constantly listen for audio information from the cloud AI using a message queue. The appropriate audio is then natively played in real-time to guide the climber, providing an immersive and interactive experience.

# Cloud Infrastructure

Riyan

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

# Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

# Acknowledgments

The team is extremely grateful to the following people for the support and mentorship throughout the project.

1. Professor Lin Weisi | Project Supervisor
2. Dr Xuan Jing | Project Co-Supervisor

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[product-screenshot]: assets/overall-solution.png
[react.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[react-url]: https://reactjs.org/
[android]: https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white
[android-url]: https://developer.android.com/
[watchos]: https://img.shields.io/badge/Apple-999999?style=for-the-badge&logo=apple&logoColor=white
[watch-url]: https://developer.apple.com/watchos/
[pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[detectron2]: https://img.shields.io/badge/Detectron2-008CBA?style=for-the-badge&logo=detectron2&logoColor=white
[detectron2-url]: https://ai.facebook.com/tools/detectron2/
[mediapipe]: https://img.shields.io/badge/Mediapipe-00A5E4?style=for-the-badge&logo=mediapipe&logoColor=white
[mediapipe-url]: https://developers.google.com/mediapipe
[aws]: https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white
[aws-url]: https://aws.amazon.com/
[mq]: https://img.shields.io/badge/AWS%20SQS-FF9900?style=for-the-badge&logo=amazon-sqs&logoColor=white
[mq-url]: https://aws.amazon.com/sqs/
[ant-media]: https://img.shields.io/badge/Ant%20Media-FF5733?style=for-the-badge&logo=ant-media-server&logoColor=white
[ant-media-url]: https://antmedia.io/
# climbAI
