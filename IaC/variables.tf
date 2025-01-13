# AWS Conf
variable "AWS_ACCESS_KEY_ID" {
  type    = string
  default = "AWS_ACCESS_KEY_ID"
}

variable "AWS_SECRET_ACCESS_KEY" {
  type    = string
  default = "AWS_SECRET_ACCESS_KEY"
}

variable "AWS_REGION" {
  description = "Region"
  #Update the below to your desired region
  default = "us-east-1"
}

variable "app_name" {
  type        = string
  description = "Application Name"
  default     = "student-health-risk"
}

variable "app_environment" {
  type        = string
  description = "Application Environment"
  default     = "production"
}

# Subnet
variable "cidr" {
  description = "The CIDR block for the VPC."
  default     = "10.32.0.0/16"
}

variable "public_subnets" {
  description = "List of public subnets"
  default     = ["10.32.100.0/24", "10.32.101.0/24"]
}

variable "private_subnets" {
  description = "List of private subnets"
  default     = ["10.32.0.0/24", "10.32.1.0/24"]
}

variable "availability_zones" {
  description = "List of availability zones"
  default     = ["us-east-1a", "us-east-1b"]
}