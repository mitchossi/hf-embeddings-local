// Helper function to check if variable is an array or Float32Array
function isArrayOrFloat32Array(variable) {
    return variable instanceof Array || variable instanceof Float32Array;
  }


export function dotProduct(a,b) {


    if( !isArrayOrFloat32Array(a) || !isArrayOrFloat32Array(b) ) {
        return "dotProduct: a and b must be arrays";
    }
    if(a.length !== b.length) return "dotProduct: a and b must be of same length";

    let dotProduct = 0;
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
    }
    return dotProduct;
}