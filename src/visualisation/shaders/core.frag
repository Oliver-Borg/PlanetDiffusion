#version 330 core


struct PointLight {
    vec3 position;
    vec3 colour;
    float strength;
};

struct Material {
    sampler2D imageTexture; // material's texture
    sampler2D normalMap;    // normal map

    float ambient;          // material's ambient lighting
    float shine;            // material's shine level
    float kd;               // diffuse reflection coefficient
    float ks;               // specular reflection coefficient
};


// input and output variables
in vec3 fragmentPosition;
in vec2 fragmentTextureCoordinate;
in vec3 fragmentNormal;

out vec4 fragColor;

uniform Material mat;

// lights
uniform PointLight lights[4];


vec3 computePointLight(PointLight light, vec3 baseTexture, vec3 fragmentPosition){

    vec3 N = normalize(fragmentNormal);
    vec3 L = normalize(light.position - fragmentPosition); 
    vec3 E = normalize(-fragmentPosition);

    // reflected direction (REPLACES halfway vector, which produced artefacts)
    vec3 R = reflect(-L, N);
    float diff = mat.kd * max(dot(L, N), 0.0);
    float spec = mat.ks * pow(max(dot(N, R), 0.0), mat.shine);

    return baseTexture * light.strength * light.colour * (diff + spec);

}

void main() {
    vec3 baseTexture = texture(mat.imageTexture, fragmentTextureCoordinate).rgb;

    // add ambient light
    vec3 result = mat.ambient * baseTexture; 

    // add effects from each point light in the scene
    for(int i = 0; i < lights.length(); ++i){
        result += computePointLight(lights[i], baseTexture, fragmentPosition);
    }

    fragColor = vec4(result, 1.0);
}
