{
  "ImageId": "ami-04505e74c0741db8d", // Ubuntu 20.04, at least in us-east-1
    "KeyName": "kernel-matrix-benchmarks", // Name of your AWS encryption key
      // Suggested instances:
      // R5b instances = GPU: None
      //                 CPU: 2nd generation Intel Xeon Scalable (Cascade Lake),
      //                      which supports AVX-512 (advanced SIMD instructions). 
      //                 + Lots of RAM.
      "InstanceType": "r5b.large", // spot price ~ $0.02/h, 2vCPU, 16Gb RAM
        // "InstanceType": "r5b.4xlarge",  // spot price ~ $0.17/h, 16vCPU, 128Gb RAM
        // "InstanceType": "r5b.16xlarge",  // spot price ~ $0.70/h, 64vCPU, 512Gb RAM
        //
        // P3 instances = GPU: Tesla V100 with 16Gb "RAM"/GPU,
        //                CPU: Intel Xeon E5-2686 v4 (Broadwell)
        //                     which supports AVX and AVX2 but not AVX-512.
        // "InstanceType": "p3.2xlarge",  // spot price ~ $0.92/h, 1GPU, 8vCPU, 61Gb RAM
        "Placement": {
    "AvailabilityZone": "us-east-1c" // North Virginia, default option
  },
  "BlockDeviceMappings": [ // "Hard drive"
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "DeleteOnTermination": false, // Just in case we want to inspect things
        "VolumeSize": 100, // 100Gb storage space
        "VolumeType": "gp2"
      }
    }
  ],
    /* The user data is a base64 encoding of the startup script below (`base64 -w 0 startup.sh`):
  
  ```bash
  #!/bin/bash
  cd /home/ubuntu
  sudo -u ubuntu git clone https://github.com/kernel-matrix-benchmarks/kernel-matrix-benchmarks.git
  cd kernel-matrix-benchmarks
  sudo -u ubuntu tmux new-session -d ./create_website_AWS.sh
  sudo -u ubuntu tmux set remain-on-exit on
  ```
  */
    "UserData": "IyEvYmluL2Jhc2gKY2QgL2hvbWUvdWJ1bnR1CnN1ZG8gLXUgdWJ1bnR1IGdpdCBjbG9uZSBodHRwczovL2dpdGh1Yi5jb20va2VybmVsLW1hdHJpeC1iZW5jaG1hcmtzL2tlcm5lbC1tYXRyaXgtYmVuY2htYXJrcy5naXQKY2Qga2VybmVsLW1hdHJpeC1iZW5jaG1hcmtzCnN1ZG8gLXUgdWJ1bnR1IHRtdXggbmV3LXNlc3Npb24gLWQgLi9jcmVhdGVfd2Vic2l0ZV9BV1Muc2gKc3VkbyAtdSB1YnVudHUgdG11eCBzZXQgcmVtYWluLW9uLWV4aXQgb24K"
}