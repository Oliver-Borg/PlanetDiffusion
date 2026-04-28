#version 330 core

layout (location=0) in vec3 vertexPosition;
layout (location=1) in vec2 vertexTextureCoordinate;
layout (location=2) in vec3 vertexNormal;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


out vec3 fragmentPosition;
out vec2 fragmentTextureCoordinate;
out vec3 fragmentNormal;


void main()
{   
    vec4 mv = model * vec4(vertexPosition, 1.0);

    fragmentTextureCoordinate = vertexTextureCoordinate;
    fragmentPosition = mv.xyz;
    fragmentNormal = normalize(mat3(model) * vertexNormal);

    gl_Position = projection * view * mv;
}
